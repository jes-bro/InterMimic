"""AdaLNEncoderLayer (learning/intermimic_transformer_network_builder.py):
at init it must equal the plain post-norm layer (unit scale, zero shift) so an
Arm A student starts as the unconditioned network; a nonzero conditioning
projection must change the output; different bodies must give different outputs.

The builder imports rl_games (unavailable locally), so the class source is
extracted and exec'd, as tests/test_exact_policy_kl.py does.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_adaln_layer.py -q
"""
import os
import re

import torch
import torch.nn as nn

FILE = os.path.join(os.path.dirname(__file__), "..", "isaacgym", "src", "intermimic",
                    "learning", "intermimic_transformer_network_builder.py")


def _load_layer():
    src = open(FILE).read()
    m = re.search(r"(class AdaLNEncoderLayer\(nn\.Module\):.*?)\n\nclass InterMimicBuilder", src, re.S)
    assert m, "AdaLNEncoderLayer not found"
    ns = {"nn": nn, "torch": torch}
    exec(m.group(1), ns)
    return ns["AdaLNEncoderLayer"]


def _plain_reference(layer, x):
    """What the layer must compute at init: post-norm attention + GELU FF, affine-free LN."""
    h = x + layer.self_attn(x, x, x, need_weights=False)[0]
    h = layer.norm1(h)
    h2 = h + layer.linear2(torch.nn.functional.gelu(layer.linear1(h)))
    return layer.norm2(h2)


def test_identity_at_init_and_body_sensitivity():
    torch.manual_seed(0)
    Layer = _load_layer()
    layer = Layer(d_model=32, nhead=4, dim_feedforward=64, cond_dim=8).eval()
    x = torch.randn(6, 5, 32)            # (tokens, batch, channels), as the builder feeds it
    cond = torch.randn(5, 8)
    with torch.no_grad():
        assert torch.allclose(layer(x, cond), _plain_reference(layer, x), atol=1e-6)   # zero-init ada = plain layer
        assert torch.allclose(layer(x, cond), layer(x, torch.zeros_like(cond)), atol=1e-6)
        nn.init.normal_(layer.ada.weight, std=0.5)                                   # once ada is live...
        out_a = layer(x, cond)
        out_b = layer(x, cond + 1.0)
    assert not torch.allclose(out_a, out_b)                                          # ...the body changes the output
    assert out_a.shape == x.shape
