# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import network_builder
from ..utils.distill_g3 import token_layout

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], : x.shape[2]]
        return x

class AdaLNEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (self-attention + GELU feed-forward,
    no dropout -- the same computation as the stock TransformerEncoderLayer used
    here) whose two LayerNorms are ADAPTIVE: their scale/shift come from a
    conditioning vector (the body embedding) instead of fixed parameters, as in
    DiT's adaLN. The conditioning projection is zero-initialized, so at init the
    layer is x -> LN(x + attn(x)) -> LN(. + ff(.)) with unit scale / zero shift,
    and the body's influence grows in during training."""

    def __init__(self, d_model, nhead, dim_feedforward, cond_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada = nn.Linear(cond_dim, 4 * d_model)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):
        # x: (T, B, C) tokens-first like the stock encoder; cond: (B, cond_dim)
        g1, b1, g2, b2 = self.ada(cond).unsqueeze(0).chunk(4, dim=-1)    # each (1, B, C)
        h = x + self.self_attn(x, x, x, need_weights=False)[0]
        h = self.norm1(h) * (1 + g1) + b1
        h2 = h + self.linear2(torch.nn.functional.gelu(self.linear1(h)))
        return self.norm2(h2) * (1 + g2) + b2


class InterMimicBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            input_shape = kwargs.pop('input_shape')[0]
            # Token layout. Historically fixed at 4 tokens ([0,1,4,16]) with the
            # encoder output read at index 1 (the delta_t=1 token). Both are now
            # params['transformer'] knobs so a 6-horizon student can exist;
            # absent = the old network exactly.
            _tf = (params.get('transformer') or {})
            # Arm A knobs (utils/body_features.py): body_dim > 0 = the trailing
            # body-feature wire (bone offsets) in the obs, consumed by adaLN in
            # every encoder layer and by the action head directly; contrastive =
            # a projection head on the trunk for the twin-env InfoNCE term.
            self._body_dim = int(_tf.get('body_dim', 0))
            self._contrastive = bool(_tf.get('contrastive', False))
            obs_per_timestep, self._num_tokens, self._readout_token = token_layout(
                input_shape, int(_tf.get('num_tokens', 4)), int(_tf.get('readout_token', 1)), self._body_dim)
            ff_size = 512
            num_channels = 256
            num_heads = 4
            self.MLPEmbedding = nn.Linear(obs_per_timestep, num_channels)
            self.PositionalEmbedding = PositionalEncoding(d_model=num_channels)
            body_emb_dim = 0
            if self._body_dim > 0:
                # Body wire: bone offsets -> 64-d embedding -> (a) adaLN scale/shift
                # in every layer, (b) concatenated into the action head. The
                # adaLN layers start as the identity (zero-init), so the network
                # begins as the plain encoder and the body modulation grows in.
                body_emb_dim = 64
                self.BodyEmbedding = nn.Sequential(nn.Linear(self._body_dim, body_emb_dim), nn.SiLU(),
                                                   nn.Linear(body_emb_dim, body_emb_dim), nn.SiLU())
                self.encoder_layers = nn.ModuleList([
                    AdaLNEncoderLayer(num_channels, num_heads, ff_size, body_emb_dim) for _ in range(3)])
                self.encoder = None
            else:
                from torch.nn import TransformerEncoderLayer
                seqTransEncoderLayer = TransformerEncoderLayer(d_model=num_channels,
                                                                    nhead=num_heads,
                                                                    dim_feedforward=ff_size,
                                                                    dropout=0,
                                                                    activation='gelu',
                                                                    batch_first=False)
                self.encoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=3)
                self.encoder = torch.compile(self.encoder)
            self._body_emb_dim = body_emb_dim
            if self._contrastive:
                self.Projection = nn.Sequential(nn.Linear(num_channels, num_channels), nn.SiLU(),
                                                nn.Linear(num_channels, int(_tf.get('proj_dim', 128))))
            self.MLPEmbedding = torch.compile(self.MLPEmbedding)
            self.PositionalEmbedding = torch.compile(self.PositionalEmbedding)

            # The transformer actor outputs num_channels (256) from the encoder, NOT
            # the actor-MLP's last unit -- so the parent A2CBuilder's mu (built at the
            # MLP width, 512) shape-mismatches it: (B,256) x (512,153). Rebuild the
            # action head(s) to take 256. fixed_sigma keeps self.sigma as a Parameter.
            actions_num = kwargs.get('actions_num')
            self.mu = nn.Linear(num_channels + body_emb_dim, actions_num)   # + the body bypass (Arm A)
            self.init_factory.create(**self.space_config['mu_init'])(self.mu.weight)
            if self.space_config.get('learn_sigma'):
                self.sigma = nn.Linear(num_channels, actions_num)
                self.init_factory.create(**self.space_config['sigma_init'])(self.sigma.weight)
            return

        def _split(self, obs):
            """(tokens [B, T, D], body [B, body_dim] or None) from the flat obs."""
            if self._body_dim > 0:
                tok, body = obs[:, :-self._body_dim], obs[:, -self._body_dim:]
                return tok.reshape(tok.shape[0], self._num_tokens, -1), body
            return obs.view(obs.shape[0], self._num_tokens, -1), None

        def trunk(self, obs):
            """The 256-d readout embedding for a flat (already normalized) obs --
            the representation the contrastive term shapes. Returns (z, body_emb)."""
            tokens, body = self._split(obs)
            body_emb = self.BodyEmbedding(body) if body is not None else None
            a_out = self.PositionalEmbedding(self.MLPEmbedding(tokens))
            a_out = a_out.permute(1, 0, 2).contiguous()          # (T, B, C)
            if self.encoder is not None:
                a_out = self.encoder(a_out)
            else:
                for layer in self.encoder_layers:
                    a_out = layer(a_out, body_emb)
            return a_out[self._readout_token], body_emb

        def project(self, z):
            """Contrastive projection of trunk outputs (only built with transformer.contrastive)."""
            return self.Projection(z)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            a_out, body_emb = self.trunk(obs)
            if body_emb is not None:
                a_out = torch.cat([a_out, body_emb], dim=-1)     # body bypass into the head

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))

                # if torch.any(~torch.isfinite(mu)):
                #     raise Exception("invalid mu")
                #     mu = torch.where(~torch.isfinite(mu), torch.zeros_like(mu), mu)
                
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                # if torch.any(~torch.isfinite(sigma)):
                #     raise Exception("invalid sigma")
                #     sigma = torch.where(~torch.isfinite(sigma), torch.zeros_like(sigma), sigma)

                return mu, sigma
            return

        def eval_critic(self, obs):
            # For transformer: obs is [batch, 6396], reshape to [batch, 4, 1599] and take first timestep
            # Embed to num_channels, then pass through critic MLP
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = InterMimicBuilder.Network(self.params, **kwargs)
        return net