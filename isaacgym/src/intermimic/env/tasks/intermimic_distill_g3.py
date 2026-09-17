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
from ...utils.body_features import body_feature_matrix, twin_pairs, twin_partners
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

        # ---- Arm A, part 1: the body wire (utils/body_features.py) ----
        # Each env's body as its rig's 52 bone offsets (156 numbers), appended
        # once to the student obs. Off by default = the plain student.
        self._body_feat_dim = 0
        self._env_body_feats = None
        if bool(env.get('studentBodyFeatures', False)):
            subj = list(env['subjectBodies'])
            paths = [str(resolve_repo_path(f"intermimic/data/assets/smplx/smplx_omomo_{s}.xml")) for s in subj]
            rows = body_feature_matrix(paths, expected_bodies=52)
            feats = torch.tensor(rows, dtype=torch.float, device=self.device)          # (n_bodies, 156)
            self._env_body_feats = feats[self._env_subject_idx.to(self.device)]        # (num_envs, 156)
            self._body_feat_dim = feats.shape[1]
            print(f"[distill-g3] body wire ON: {len(subj)} rigs -> {self._body_feat_dim}-d bone-offset "
                  f"features per env (studentBodyFeatures)", flush=True)

        # ---- student observation layout, derived from the teacher's ----
        self._student_horizons = validate_horizons(env['studentObsHorizons'], 'studentObsHorizons')
        expected = student_obs_width(self.obs_buf.shape[1], self._obs_horizons,
                                     self._student_horizons, bool(getattr(self, '_use_betas_obs', False)),
                                     extra_dims=self._body_feat_dim)
        if int(env['numObsRetarget']) != expected:
            raise ValueError(
                f"[distill-g3] numObsRetarget {env['numObsRetarget']} != {expected} = "
                f"{len(self._student_horizons)} student horizons x "
                f"{self.obs_buf.shape[1] // len(self._obs_horizons)} per horizon + "
                f"{self._body_feat_dim} body-feature dims "
                f"(teacher numObs {self.obs_buf.shape[1]} over obsHorizons {self._obs_horizons})")

        # ---- Arm A, part 2: twin envs for the contrastive term ----
        # Pairs (e, e + n_objects): same object bucket, different body. The second
        # env of a pair copies the first's clip + start frame at every reset
        # (_twin_sync), so a pair is the SAME motion on TWO bodies until one of
        # them terminates; twin_valid tracks that per step. Off by default.
        self._twin_envs = bool(env.get('twinEnvs', False))
        self.twin_a = self.twin_b = None
        self.twin_valid = None
        if self._twin_envs:
            partner = twin_partners(self.num_envs, len(self.object_name), len(env['subjectBodies']))
            pairs = twin_pairs(partner)
            if not pairs:
                raise ValueError("[distill-g3] twinEnvs: no pairs possible (num_envs too small?)")
            self.twin_a = torch.tensor([a for a, _ in pairs], device=self.device, dtype=torch.long)
            self.twin_b = torch.tensor([b for _, b in pairs], device=self.device, dtype=torch.long)
            self._twin_partner = torch.tensor(partner, device=self.device, dtype=torch.long)
            self.twin_valid = torch.zeros(len(pairs), dtype=torch.bool, device=self.device)
            print(f"[distill-g3] twin envs ON: {len(pairs)} pairs over {self.num_envs} envs "
                  f"({2 * len(pairs)} paired, {self.num_envs - 2 * len(pairs)} unpaired; "
                  f"partner = e + {len(self.object_name)} objects)", flush=True)
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

    # ------------------------------------------------------------------ twins
    def _twin_sync(self, env_ids, motion_ids, motion_times, ref_idx):
        """Called by the parent's reset paths with the freshly sampled (motion,
        start frame, PSI slot) for env_ids. For every twin pair with BOTH envs in
        this reset batch, the second env takes the first's CLIP (mapped into its
        own body block) and start frame, and re-samples its PSI slot for that
        motion (the slot it drew was for a different motion; copying the first's
        slot could point at a slot never written for this body's motion)."""
        if not self._twin_envs:
            return motion_ids, motion_times, ref_idx
        pos = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        pos[env_ids] = torch.arange(env_ids.shape[0], device=self.device)
        pa, pb = pos[self.twin_a], pos[self.twin_b]
        both = (pa >= 0) & (pb >= 0)
        if not bool(both.any()):
            return motion_ids, motion_times, ref_idx
        pa, pb = pa[both], pb[both]
        clip = motion_ids[pa] % self._n_clips
        motion_ids[pb] = clip + self._env_subject_idx[self.twin_b[both]] * self._n_clips
        motion_times[pb] = motion_times[pa]
        rr = self.ref_reward[motion_ids[pb], :, motion_times[pb]]
        cdf = torch.cumsum(rr / rr.sum(1, keepdim=True), dim=1)
        ref_idx[pb] = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1), device=cdf.device)).squeeze(1)
        return motion_ids, motion_times, ref_idx

    def _update_twin_valid(self):
        """A pair is a positive only while both envs are on the same clip at the
        same frame (a termination on either side desyncs them until both reset)."""
        if not self._twin_envs:
            return
        same_clip = (self.data_id[self.twin_a] % self._n_clips) == (self.data_id[self.twin_b] % self._n_clips)
        same_t = self.progress_buf[self.twin_a] == self.progress_buf[self.twin_b]
        self.twin_valid = same_clip & same_t

    # ------------------------------------------------------------------ student obs
    def _compute_observations(self, env_ids=None):
        # Parent fills obs_buf (teacher obs) and _curr_ref_obs from the per-body
        # reference; the student stacks the SAME reference over its own horizons,
        # plus (Arm A) its body-feature wire.
        super()._compute_observations(env_ids)
        if not self._g3_ready:
            return
        if env_ids is None:
            stacked = self._stack_obs_horizons(None, self._student_horizons, None)
            if self._env_body_feats is not None:
                stacked = torch.cat([stacked, self._env_body_feats], dim=-1)
            self.obs_buf_retarget[:] = stacked
        else:
            stacked = self._stack_obs_horizons(env_ids, self._student_horizons, None)
            if self._env_body_feats is not None:
                stacked = torch.cat([stacked, self._env_body_feats[env_ids]], dim=-1)
            self.obs_buf_retarget[env_ids] = stacked
        self._update_twin_valid()

    # ------------------------------------------------------------------ teacher query
    def single_model_forward(self, params, obs, mean, var):
        curr_obs = torch.clamp((obs - mean) / torch.sqrt(var + 1e-5), min=-5.0, max=5.0)
        res = self.functional_models[0](params, {'is_train': False, 'prev_actions': None,
                                                 'obs': curr_obs, 'rnn_states': None})
        return res['mus'], res['sigmas']

    def _query_teachers(self):
        with torch.no_grad():
            batched = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            n = self.running_means_all.shape[0]
            mus_all, sigma_all = batched(self.stacked_params,
                                         self.obs_buf.unsqueeze(0).repeat(n, 1, 1),
                                         self.running_means_all, self.running_vars_all)
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
