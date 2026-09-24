"""InterMimicDistillG3 -- DAgger distillation of g3 teachers into one student.

The g3 teachers (OMOMO per-source, bball7, soccer15, cpr13) were all trained by
InterMimic with per-body contact-retargeted references (retargetedMotionDir),
obsHorizons [1,4,7,10,13,16] (numObs 9594), cpuMotionData, optionally ragged
storage. This task is that SAME parent with two additions and nothing else:

  * a student observation, obs_buf_retarget, built from the same per-body
    reference the teacher sees, over `studentObsHorizons` (any list; the MLP
    student uses the teacher's six, the transformer student too -- 6 tokens);
  * a teacher query after every physics step and every reset: each env's
    action label comes from the teacher that owns its clip's SOURCE subject
    (source_subject_index[data_id]), chosen via teachers.yaml in teacherPolicy.

Everything the teacher pipeline does (reference load, ragged/cpu motion,
PSI harvest in _compute_reset, free-flight gate, termination logging) runs
unchanged because step()/post_physics_step() are the parent's -- only the two
hooks above are appended. Compare InterMimic_All, which re-implements the
loop, queries the teacher on the SOURCE mocap, hardcodes the student horizons,
and routes by the TARGET body (see utils/distill_g3.py for the details).

Fails at startup (not at epoch 1) on: missing/invalid teachers.yaml, a source
in the data with no teacher, a teacher whose input width is not this env's
numObs, numObsRetarget not matching studentObsHorizons, betas conditioning.

Env cfg keys (on top of a g3 teacher env cfg):
  teacherPolicy:      dir holding the teacher .pth files + teachers.yaml
  teacherPolicyCFG:   the teachers' rl_games train yaml (network arch)
  studentObsHorizons: delta_t list for the student obs
  numObsRetarget:     len(studentObsHorizons) * (numObs / len(obsHorizons))
"""
import os

import torch
import yaml
from functorch import make_functional
from torch.func import vmap

from rl_games.algos_torch import torch_ext

from .intermimic import InterMimic
from ...learning import (intermimic_models_teacher, intermimic_network_builder,
                         intermimic_transformer_network_builder)
from ...utils.distill_g3 import (build_source_lookup, load_teacher_manifest,
                                 student_obs_width, validate_horizons)
from ...utils.path_utils import resolve_data_path, resolve_repo_path


class InterMimicDistillG3(InterMimic):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Construction-time guard: the parent's __init__ may run reset() /
        # _compute_observations() before our buffers exist. Until _g3_ready is
        # set at the END of this __init__, the overrides below only call super.
        self._g3_ready = False
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        env = cfg['env']
        for k in ('teacherPolicy', 'teacherPolicyCFG', 'studentObsHorizons', 'numObsRetarget'):
            if k not in env:
                raise KeyError(f"[distill-g3] env cfg lacks required key '{k}'")

        # ---- student observation layout, derived from the teacher's ----
        self._student_horizons = validate_horizons(env['studentObsHorizons'], 'studentObsHorizons')
        expected = student_obs_width(self.obs_buf.shape[1], self._obs_horizons,
                                     self._student_horizons, bool(getattr(self, '_use_betas_obs', False)))
        if int(env['numObsRetarget']) != expected:
            raise ValueError(
                f"[distill-g3] numObsRetarget {env['numObsRetarget']} != {expected} = "
                f"{len(self._student_horizons)} student horizons x "
                f"{self.obs_buf.shape[1] // len(self._obs_horizons)} per horizon "
                f"(teacher numObs {self.obs_buf.shape[1]} over obsHorizons {self._obs_horizons})")
        self.obs_buf_retarget = torch.zeros((self.num_envs, expected), device=self.device, dtype=torch.float)
        self.action_buf = torch.zeros((self.num_envs, 153), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros((self.num_envs, 153), device=self.device, dtype=torch.float)
        print(f"[distill-g3] teacher obs {self.obs_buf.shape[1]} over horizons {self._obs_horizons}; "
              f"student obs {expected} over horizons {self._student_horizons}", flush=True)

        # ---- teacher network (one architecture for the whole fleet) ----
        teacher_cfg_path = resolve_data_path("cfg", "train", "rlg", os.path.basename(env['teacherPolicyCFG']))
        with open(teacher_cfg_path) as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)
        net_name = cfg_teacher['params']['network']['name']
        if net_name == 'intermimic_transformer':
            network = intermimic_transformer_network_builder.InterMimicBuilder()
        elif net_name == 'intermimic':
            network = intermimic_network_builder.InterMimicBuilder()
        else:
            raise ValueError(f"[distill-g3] unsupported teacher network '{net_name}' in {teacher_cfg_path}")
        network.load(cfg_teacher['params']['network'])
        network = intermimic_models_teacher.ModelInterMimicContinuous(network)
        build_cfg = {
            'actions_num': 153,
            'input_shape': (self.obs_buf.shape[1],),
            'num_seqs': self.num_envs,
            'value_size': 1,
        }

        # ---- teacher checkpoints, per the manifest ----
        teacher_dir = resolve_repo_path(env['teacherPolicy'])
        self._teacher_entries = load_teacher_manifest(teacher_dir)
        params_list, means, vars_ = [], [], []
        self.functional_models = []
        for e in self._teacher_entries:
            ck = torch_ext.load_checkpoint(os.path.join(teacher_dir, e.file))
            model = network.build(build_cfg).to(self.device)
            model.load_state_dict(ck['model'])
            rm = ck['running_mean_std']['running_mean']
            if tuple(rm.shape) != (self.obs_buf.shape[1],):
                raise ValueError(
                    f"[distill-g3] teacher {e.file} was trained on obs width {tuple(rm.shape)} but "
                    f"this env's numObs is {self.obs_buf.shape[1]} (obsHorizons {self._obs_horizons}); "
                    f"the teacher would be queried off-distribution")
            f_model, params = make_functional(model)
            self.functional_models.append(f_model)
            params_list.append(params)
            means.append(rm)
            vars_.append(ck['running_mean_std']['running_var'])
            print(f"[distill-g3] teacher {e.file}: sources {e.sources}, epoch {e.epoch}, from {e.origin}", flush=True)
        self.stacked_params = tuple(torch.stack(p, dim=0) for p in zip(*params_list))
        self.running_means_all = torch.stack(means).float().to(self.device)
        self.running_vars_all = torch.stack(vars_).float().to(self.device)

        # ---- source -> teacher routing, checked against the data at startup ----
        present = torch.unique(self.source_subject_index).tolist()
        lookup, unused = build_source_lookup(self._teacher_entries, present)
        self._source_to_model = torch.tensor(lookup, device=self.device, dtype=torch.long)
        if unused:
            print(f"[distill-g3] NOTE: {len(unused)} teacher(s) serve no clip in this data: {unused}", flush=True)
        print(f"[distill-g3] {len(self._teacher_entries)} teachers cover {len(present)} sources: "
              f"{['sub%d' % s for s in present]}", flush=True)

        self._g3_ready = True
        self._refresh_teacher_indices()

    # ------------------------------------------------------------------ routing
    def _refresh_teacher_indices(self):
        """Per-env teacher index from the SOURCE subject of the env's current clip.
        Called on every reset (data_id changes there) and by the distill agent."""
        src = self.source_subject_index[self.data_id]
        idx = self._source_to_model[src]
        if bool((idx < 0).any()):
            bad = torch.unique(src[idx < 0]).tolist()
            raise RuntimeError(f"[distill-g3] envs routed to sources with no teacher: {bad}")
        self.model_indices = idx
        self.sample_indices = torch.arange(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ student obs
    def _compute_observations(self, env_ids=None):
        # Parent fills obs_buf (teacher obs) and _curr_ref_obs from the per-body
        # reference; the student stacks the SAME reference over its own horizons.
        super()._compute_observations(env_ids)
        if not self._g3_ready:
            return
        if env_ids is None:
            self.obs_buf_retarget[:] = self._stack_obs_horizons(None, self._student_horizons, None)
        else:
            self.obs_buf_retarget[env_ids] = self._stack_obs_horizons(env_ids, self._student_horizons, None)
        self._sanitize_student_obs(env_ids)

    def _sanitize_student_obs(self, env_ids=None):
        """Zero non-finite entries in the STUDENT's observation buffer.

        WHY. humanoid.compute_humanoid_reset already does exactly this for
        obs_buf, and its comment says what for: "the policy's next forward pass
        would crash on these inputs (Normal(loc=NaN) raises ValueError). Zero is
        finite and the policy can produce a valid (garbage) action that gets
        discarded on reset." That guard was written when obs_buf WAS the policy's
        input. This task feeds the policy obs_buf_retarget instead, and the guard
        never followed it -- so the teacher path is protected and the student
        path is not. This restores that invariant ("what the policy is fed is
        finite") for the buffer the policy actually reads; it is not a new
        policy, and declining it here while every teacher eval has it would make
        the student's numbers less comparable to the teachers', not more.

        The envs concerned are already being terminated by the same reset logic
        that flagged the NaN, so the action taken from a zeroed observation is
        discarded. Without this, three of the 169 xf@29k in-dist pairs (bodies
        sub12/sub14/sub17, all against source sub8) died in rl_games'
        models.py:243 distr.sample() and scored nothing at all.

        Loud once, like the teacher guard: a student input going non-finite is
        worth seeing, and silence here would hide a real numerical problem.
        """
        buf = self.obs_buf_retarget if env_ids is None else self.obs_buf_retarget[env_ids]
        bad = ~torch.isfinite(buf)
        if not bool(bad.any()):
            return
        n_env = int(torch.any(bad, dim=-1).sum())
        if not getattr(self, '_student_obs_nan_warned', False):
            self._student_obs_nan_warned = True
            print(f"[distill_g3] WARNING: non-finite student observation in {n_env} "
                  f"env(s) ({int(bad.sum())} of {bad.numel()} values); zeroing, as "
                  f"humanoid.py does for obs_buf. Those envs are terminated by the "
                  f"reset logic, so the action is discarded. Further occurrences "
                  f"are not reported.", flush=True)
        if env_ids is None:
            self.obs_buf_retarget[bad] = 0.0
        else:
            buf[bad] = 0.0
            self.obs_buf_retarget[env_ids] = buf

    # ------------------------------------------------------------------ teacher query
    def single_model_forward(self, params, obs, mean, var):
        curr_obs = torch.clamp((obs - mean) / torch.sqrt(var + 1e-5), min=-5.0, max=5.0)
        res = self.functional_models[0](params, {'is_train': False, 'prev_actions': None,
                                                 'obs': curr_obs, 'rnn_states': None})
        return res['mus'], res['sigmas']

    # Smallest sigma the teacher sampler is allowed. Only ever applied to
    # entries that were already NaN/inf or negative -- see _sanitize_teacher.
    _TEACHER_SIGMA_FLOOR = 1e-4

    def _sanitize_teacher(self, mus_all, sigma_all):
        """Repair a NaN/inf teacher output instead of crashing the whole job.

        WHY. An env whose observation goes numerically invalid feeds NaN into
        every teacher, so mus/sigmas come back NaN and
        Normal(mus, sigma).sample() raises "normal expects all elements of
        std >= 0.0" -- killing all 16 envs of an eval over a handful of bad
        ones (4 of 169 pairs in the xf@29k in-dist matrix, all source sub8).
        The humanoid already handles the same event one level up ("invalid
        observation in N env(s); terminating those envs and continuing"); this
        is that policy applied to the teacher query.

        This cannot change a healthy run. It only touches entries that are
        NaN, inf or negative, and every one of those is a value the sampler
        would have raised on -- so the previous behaviour for them was a
        crash, not a number. At EVAL it cannot move a metric at all, because
        the player discards the teacher action (intermimic_players_distill.py
        env_reset returns obs only).

        Loud, not silent: the first repair prints what it fixed and how much,
        so a teacher that has genuinely gone bad is visible rather than
        quietly sampled around.
        """
        bad_mu = ~torch.isfinite(mus_all)
        bad_sigma = ~torch.isfinite(sigma_all) | (sigma_all < 0)
        n_mu, n_sigma = int(bad_mu.sum()), int(bad_sigma.sum())
        if n_mu == 0 and n_sigma == 0:
            return mus_all, sigma_all
        if not getattr(self, '_teacher_nan_warned', False):
            self._teacher_nan_warned = True
            print(f"[distill_g3] WARNING: teacher query produced {n_mu} non-finite "
                  f"mu and {n_sigma} invalid sigma value(s) out of {mus_all.numel()}; "
                  f"repairing (mu->0, sigma->{self._TEACHER_SIGMA_FLOOR}) and "
                  f"continuing. This follows an invalid observation upstream, and "
                  f"the teacher action is unused at eval. Further occurrences are "
                  f"not reported.", flush=True)
        mus_all = torch.nan_to_num(mus_all, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_all = torch.nan_to_num(sigma_all, nan=self._TEACHER_SIGMA_FLOOR,
                                     posinf=self._TEACHER_SIGMA_FLOOR,
                                     neginf=self._TEACHER_SIGMA_FLOOR)
        return mus_all, sigma_all.clamp_min(self._TEACHER_SIGMA_FLOOR)

    def _query_teachers(self):
        with torch.no_grad():
            batched = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            n = self.running_means_all.shape[0]
            mus_all, sigma_all = batched(self.stacked_params,
                                         self.obs_buf.unsqueeze(0).repeat(n, 1, 1),
                                         self.running_means_all, self.running_vars_all)
            mus_all, sigma_all = self._sanitize_teacher(mus_all, sigma_all)
            act = torch.clamp(torch.distributions.Normal(mus_all, sigma_all).sample(), -1.0, 1.0)
            self.action_buf = act[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]

    def post_physics_step(self):
        super().post_physics_step()
        if self._g3_ready:
            self._query_teachers()

    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        if self._g3_ready:
            self._refresh_teacher_indices()
            self._query_teachers()
