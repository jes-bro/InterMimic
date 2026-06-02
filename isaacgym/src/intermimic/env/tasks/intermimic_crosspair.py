"""Cross-pair distillation env.

Each env is assigned a (body, source, object) triple and supervised by the
specific teacher trained on that exact triple. Different from InterMimic_All
which keys teachers by a single subject_id — here the lookup is per-triple
because our 16 teachers cover (4 bodies) x (2 sources) x (2 objects).

Teacher checkpoint files must be named with the cross-pair slug:
    b<body>_s<source>_<object>.pth
e.g.:
    b10_s2_largetable.pth
    b17_s6_woodchair.pth

The student uses the same observation as the teachers (3230-dim with betas
conditioning) — no separate retarget obs. Both teacher and student see
`(source motion, target body betas)` and the teacher's action is the BC
target for the student's policy at that env.
"""

import os
import re
import yaml
import torch
import numpy as np
from isaacgym.torch_utils import to_torch
from rl_games.algos_torch import torch_ext
from torch.func import vmap
from functorch import make_functional

from .intermimic import InterMimic
from ...learning import intermimic_network_builder, intermimic_models_teacher
from ...utils.path_utils import resolve_data_path, resolve_repo_path

# Teacher checkpoint filename slug: b<body>_s<source>_<object>.pth
TEACHER_SLUG_RE = re.compile(r'^b(\d+)_s(\d+)_(.+)\.pth$')


def _parse_teacher_slug(filename):
    basename = os.path.basename(filename)
    m = TEACHER_SLUG_RE.match(basename)
    if m is None:
        raise ValueError(f"Could not parse teacher slug from {basename!r}; "
                         f"expected pattern b<body>_s<source>_<object>.pth")
    return int(m.group(1)), int(m.group(2)), m.group(3)


class InterMimic_CrossPair(InterMimic):
    """Multi-teacher distillation env keyed by (body, source, object)."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
            device_type=device_type, device_id=device_id, headless=headless,
        )

        # Distillation buffers (read by VecTaskDAggerWrapper)
        self.action_buf = torch.zeros((self.num_envs, 153), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros((self.num_envs, 153), device=self.device, dtype=torch.float)
        # Student obs: by default same 3230-dim obs as teachers. For the
        # no-betas student ablation, set numObsRetarget = numObs - 32 (3198)
        # in cfg — the betas channel (last 32 dims) gets stripped before the
        # student sees its obs.
        num_obs_full = cfg["env"]["numObs"]
        num_obs_retarget = cfg["env"].get("numObsRetarget", num_obs_full)
        if num_obs_retarget == num_obs_full:
            self.obs_buf_retarget = self.obs_buf  # alias, same tensor
            self._strip_betas_for_student = False
        elif num_obs_retarget == num_obs_full - 32:
            self.obs_buf_retarget = torch.zeros(
                (self.num_envs, num_obs_retarget),
                device=self.device, dtype=torch.float,
            )
            self._strip_betas_for_student = True
        else:
            raise ValueError(
                f"numObsRetarget ({num_obs_retarget}) must equal numObs "
                f"({num_obs_full}) for same-obs student, or numObs-32 "
                f"({num_obs_full - 32}) for the no-betas student ablation."
            )

        teacher_dir = resolve_repo_path(cfg["env"]["teacherPolicy"])
        teacher_files = sorted(
            os.path.join(teacher_dir, f)
            for f in os.listdir(teacher_dir)
            if f.endswith('.pth')
        )
        if not teacher_files:
            raise FileNotFoundError(f"No .pth teachers found in {teacher_dir}")

        teacher_cfg_path = resolve_data_path(
            "cfg", "train", "rlg",
            os.path.basename(cfg["env"]["teacherPolicyCFG"]),
        )
        with open(teacher_cfg_path, 'r') as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)

        net_builder = intermimic_network_builder.InterMimicBuilder()
        net_builder.load(cfg_teacher['params']['network'])
        net_template = intermimic_models_teacher.ModelInterMimicContinuous(net_builder)

        obs_shape = cfg["env"]["numObs"]
        model_config = {
            'actions_num': 153,
            'input_shape': (obs_shape,),
            'num_seqs': self.num_envs,
            'value_size': 1,
        }

        self.teacher_triples = []
        self.functional_models = []
        self.params_list = []
        self.running_means = []
        self.running_vars = []
        for path in teacher_files:
            triple = _parse_teacher_slug(path)
            self.teacher_triples.append(triple)
            ck = torch_ext.load_checkpoint(path)
            model = net_template.build(model_config)
            model.to(self.device)
            model.load_state_dict(ck['model'])
            f_model, params = make_functional(model)
            self.functional_models.append(f_model)
            self.params_list.append(params)
            self.running_means.append(ck['running_mean_std']['running_mean'])
            self.running_vars.append(ck['running_mean_std']['running_var'])

        # Stack per-teacher params + stats along a new model dim for vmap
        params_zip = list(zip(*self.params_list))
        self.stacked_params = tuple(torch.stack(p_tensors, dim=0) for p_tensors in params_zip)
        self.running_means_all = torch.stack(self.running_means).float()
        self.running_vars_all = torch.stack(self.running_vars).float()

        # (body, source, obj) -> teacher index
        self.teacher_lookup = {t: i for i, t in enumerate(self.teacher_triples)}
        print(f"[InterMimic_CrossPair] Loaded {len(self.teacher_triples)} teachers:")
        for i, t in enumerate(self.teacher_triples):
            print(f"  [{i}] body=sub{t[0]} source=sub{t[1]} object={t[2]}")

        # Optional excludeCombos: list of [body, source, obj] triples that
        # have no teacher (e.g., training was unstable). When an env gets
        # routed to one of these triples, we use a FALLBACK teacher that
        # shares the same body and object but a different source. The
        # student still gets body-appropriate BC supervision for that env,
        # just from a teacher trained on a different source motion.
        exclude_cfg = cfg["env"].get("excludeCombos", []) or []
        self.exclude_combos = set()
        self.fallback_lookup = {}
        for entry in exclude_cfg:
            triple = (int(entry[0]), int(entry[1]), str(entry[2]))
            self.exclude_combos.add(triple)
            # Find a fallback: same body and object, any other available source.
            body, source, obj = triple
            fallback = None
            for t in self.teacher_triples:
                if t[0] == body and t[2] == obj and t[1] != source:
                    fallback = t
                    break
            if fallback is None:
                # No same-body fallback. Try same body any object.
                for t in self.teacher_triples:
                    if t[0] == body:
                        fallback = t
                        break
            if fallback is None:
                raise ValueError(
                    f"excludeCombo {triple} has no usable fallback teacher "
                    f"(no teacher exists for body=sub{body})."
                )
            self.fallback_lookup[triple] = self.teacher_lookup[fallback]
            print(f"[InterMimic_CrossPair] excludeCombo {triple} -> fallback "
                  f"teacher body=sub{fallback[0]} source=sub{fallback[1]} "
                  f"object={fallback[2]} (idx {self.teacher_lookup[fallback]})")

        self.model_indices = None
        self.sample_indices = None
        # Per-env mask: True for envs whose current triple is in exclude_combos.
        # Used by the distillation agent to mask out those envs' contributions
        # to BC, actor, critic, entropy, and bounds losses (so the student
        # never learns from triples we know are physically infeasible /
        # whose teacher training was unstable).
        self.excluded_env_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        # Populate model_indices + excluded_env_mask immediately so the
        # distillation agent's first read of these (before any env.step())
        # reflects the real current triples, not the all-False default.
        # Base InterMimic.__init__ already assigned data_id/dataset_id, so
        # _compute_env_triples will return real values here.
        self._refresh_teacher_indices()
        return

    def _compute_env_triples(self, env_ids):
        """Return list of (body_num, source_num, obj_name) for the given env indices."""
        triples = []
        for env_i in env_ids:
            env_i = int(env_i)
            body_sub_str = self.subject_bodies[int(self._env_subject_idx[env_i])]
            body_num = int(body_sub_str[3:])
            source_num = int(self.dataset_id[env_i])
            motion_idx = int(self.data_id[env_i])
            obj_name = self.object_name[int(self.object_id[motion_idx])]
            triples.append((body_num, source_num, obj_name))
        return triples

    def _refresh_teacher_indices(self):
        """Recompute model_indices for ALL envs based on their current triples.
        Excluded triples (in self.exclude_combos) get routed to a fallback teacher
        for env stepping, and self.excluded_env_mask is set True for those envs
        so the distillation agent can mask them out of all loss terms.
        """
        triples = self._compute_env_triples(range(self.num_envs))

        indices = []
        excluded_flags = []
        for t in triples:
            if t in self.teacher_lookup:
                indices.append(self.teacher_lookup[t])
                excluded_flags.append(False)
            elif t in self.fallback_lookup:
                # No teacher for this triple. Fallback (same body, diff source)
                # keeps the env physically stable; agent masks loss to 0.
                indices.append(self.fallback_lookup[t])
                excluded_flags.append(True)
            else:
                raise KeyError(
                    f"Env triple {t} has no teacher and no fallback. "
                    f"Available teachers: {sorted(self.teacher_lookup.keys())}, "
                    f"excludeCombos: {sorted(self.exclude_combos)}"
                )

        self.model_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.sample_indices = torch.arange(self.num_envs, device=self.device)
        self.excluded_env_mask = torch.tensor(
            excluded_flags, dtype=torch.bool, device=self.device,
        )
        return

    def _single_model_forward(self, params, obs, mean, var):
        norm_obs = (obs - mean) / torch.sqrt(var + 1e-5)
        norm_obs = torch.clamp(norm_obs, min=-5.0, max=5.0)
        input_dict = {
            'is_train': False, 'prev_actions': None,
            'obs': norm_obs, 'rnn_states': None,
        }
        res = self.functional_models[0](params, input_dict)
        return res['mus'], res['sigmas']

    def _query_all_teachers(self):
        """Run all teachers via vmap on obs_buf; pick per-env action via model_indices."""
        with torch.no_grad():
            batched_forward = vmap(self._single_model_forward, in_dims=(0, 0, 0, 0))
            obs_batched = self.obs_buf.unsqueeze(0).expand(self.running_means_all.shape[0], -1, -1)
            mus_all, sigma_all = batched_forward(
                self.stacked_params, obs_batched, self.running_means_all, self.running_vars_all,
            )
            distr = torch.distributions.Normal(mus_all, sigma_all)
            sampled_all = torch.clamp(distr.sample(), min=-1.0, max=1.0)
            self.action_buf = sampled_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]
        return

    def _sync_student_obs(self):
        """Refresh obs_buf_retarget from obs_buf. For the no-betas student
        ablation, strips the trailing 32-dim betas channel."""
        if self._strip_betas_for_student:
            self.obs_buf_retarget[:] = self.obs_buf[:, :-32]
        # else: obs_buf_retarget is the same tensor as obs_buf, no copy needed

    def step(self, actions):
        super().step(actions)
        # Refresh on EVERY step. Envs that just reset may have new motions
        # → new (body, source, object) triples → mask + routing must update.
        # Old code only refreshed once at startup, leaving stale routing
        # after any env reset.
        self._refresh_teacher_indices()
        self._query_all_teachers()
        self._sync_student_obs()
        return

    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        # On reset, env body / source / object assignments may have changed,
        # so re-look-up teacher indices and re-query for the new state.
        self._refresh_teacher_indices()
        self._query_all_teachers()
        self._sync_student_obs()
        return
