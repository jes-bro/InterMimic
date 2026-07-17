from enum import Enum
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from ...utils import torch_utils
import torch.nn.functional as F
from .humanoid import *
import trimesh
import imageio

from ...utils.path_utils import resolve_data_path


# ----------------------------------------------------------------------------
# Cross-body filename parsing helpers
# ----------------------------------------------------------------------------
# Motion files come in two naming conventions:
#   * Identity (standard InterMimic): 'sub<N>_<obj>_<idx>.pt'
#       Meaning: subject N's body performs the motion. Source body == target body.
#   * Cross-body (new, this project):  'sub<src>to<tgt>_<obj>_<idx>.pt'
#       Meaning: subject <src>'s motion has been retargeted onto subject <tgt>'s
#       body shape by the cross-body retargeter. The sim controls subject <tgt>;
#       the reference motion came from subject <src>.
#
# The env needs to know both numbers so it can (a) load the right MJCF for the
# target body, and (b) feed (source_betas, target_betas) into the observation.

def _parse_motion_subject(filename):
    """Parse a motion filename and return (source_subject_num, target_subject_num).

    Examples:
        'sub2_largetable_007.pt'      -> (2, 2)    # identity pair
        'sub2to8_largetable_007.pt'   -> (2, 8)    # cross-body
        '/path/to/sub3to11_box_001.pt' -> (3, 11)  # works on full paths too

    Returns ints. Raises ValueError on malformed input.
    """
    first_token = os.path.basename(filename).split('_')[0]
    if not first_token.startswith('sub'):
        raise ValueError(
            "Motion filename must start with 'sub<N>_...' or 'sub<src>to<tgt>_...': "
            "{}".format(filename))
    body = first_token[3:]   # strip 'sub' prefix -> '2', '2to8', etc.
    if 'to' in body:
        src_str, tgt_str = body.split('to', 1)
        return int(src_str), int(tgt_str)
    return int(body), int(body)


def _load_betas_npz(npz_path):
    """Load the per-subject betas lookup produced by extract_omomo_betas.py.

    Returns a dict keyed by subject NUMBER (int) -> (16,) np.ndarray of betas.
    The on-disk format keys by 'sub<N>' strings; we convert to int keys here
    because that's what the env code does index lookups against.
    """
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for key in data.files:
        if key.startswith('sub') and key[3:].isdigit():
            out[int(key[3:])] = np.asarray(data[key], dtype=np.float32)
    if not out:
        raise ValueError("No 'sub<N>' keys in betas npz {}".format(npz_path))
    return out


# Per-subject body heights (m) from measure_subject_bodies.py on the
# generated per-subject MJCFs. Used by the bodyNormalizedReward feature
# to remove the body-size bias in InterMimic's default world-frame pose-
# error reward (smaller bodies look like they have proportionally more error).
SUBJECT_HEIGHTS = {
    1: 1.451, 2: 1.601, 3: 1.692, 4: 1.546, 5: 1.576,
    6: 1.528, 7: 1.572, 8: 1.594, 9: 1.538, 10: 1.437,
    11: 1.528, 12: 1.554, 13: 1.562, 14: 1.581, 15: 1.469,
    16: 1.489, 17: 1.496,
}


class InterMimic(Humanoid_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    # Every key the task recognizes under cfg['env'] (union of all committed configs
    # + every cfg['env'] access in the task code). A key NOT here is almost certainly
    # a TYPO (e.g. 'subjectBody', 'bodyNormalisedReward') that would otherwise be
    # silently ignored -> the run quietly does the wrong thing (feature off, etc.).
    KNOWN_ENV_KEYS = frozenset({
        'asset', 'ballSize', 'betas_file', 'bodyNormalizedReward', 'contactBodies',
        'contactIndex', 'controlFrequencyInv', 'cpuMotionData', 'dataFPS',
        'dataFramesScale', 'dataObjects', 'dataSub', 'enableDebugVis',
        'enableEarlyTermination', 'enableEvaluation', 'envSpacing', 'episodeLength',
        'excludeCombos', 'hybridInitProb', 'initRootHeight', 'initVel', 'isFlagrun',
        'keyBodies', 'keyIndex', 'localRootObs', 'maskDeadEnvs', 'maxClipsPerObject',
        'moreRigid', 'motion_file', 'motion_file_retarget', 'numActions', 'numDoF',
        'numDoFHand', 'numDoFWrist', 'numEnvs', 'numObs', 'numObsRetarget',
        'numObservations', 'numStates', 'objectDensity', 'pairSampleCountsFile',
        'pdControl', 'physicalBufferSize', 'plane', 'playdataset', 'powerScale',
        'projtype', 'rewardTerms', 'rewardWeights', 'robotType', 'rolloutLength',
        'rootHeightObs', 'saveImages', 'scaling', 'stateInit', 'subjectBodies',
        'subjectHeightsFile', 'subjectPairWeightsFile', 'teacherPolicy',
        'teacherPolicyCFG', 'terminationHeight', 'useTransformerObs',
        # 'seed' is injected into cfg['env'] by rl_games' player on the --test
        # path (NOT training), so it's a legitimate runtime key, not a typo.
        'seed',
        # objectAug (physics): per-env object scale/yaw/translate + mass correction.
        # objectTermsEnable gates the stock object-match reward terms (ro*rig*rcg),
        # which a perturbed object makes unachievable.
        'objectAug', 'objectTermsEnable',
    })

    def _validate_env_config(self, env_cfg):
        """Fail loudly on an unrecognized (typo'd) env or reward-term key rather than
        silently ignoring it and running the wrong experiment (no-fallback policy)."""
        unknown = sorted(k for k in env_cfg if k not in self.KNOWN_ENV_KEYS)
        if unknown:
            raise ValueError(
                f"[intermimic] unrecognized env config key(s): {unknown}. Likely a "
                f"TYPO -- a misspelled key is silently ignored and the run does the "
                f"wrong thing. If a key is genuinely new, add it to "
                f"InterMimic.KNOWN_ENV_KEYS.")
        rt = env_cfg.get('rewardTerms') or {}
        # 'pose' (relative joint-angle) and 'hold' (objectAug relaxed-contact) are the
        # two opt-in reward terms; each takes only {enable, lambda}. A key outside this
        # set is almost certainly a typo that would silently disable the term.
        bad = sorted(k for k in rt if k not in ('pose', 'hold'))
        if bad:
            raise ValueError(f"[intermimic] unknown rewardTerms key(s) {bad} "
                             f"(only 'pose', 'hold').")
        for _term in ('pose', 'hold'):
            badp = sorted(k for k in (rt.get(_term) or {}) if k not in ('enable', 'lambda'))
            if badp:
                raise ValueError(f"[intermimic] unknown rewardTerms.{_term} key(s) {badp} "
                                 f"(only 'enable', 'lambda').")
        # objectAug sub-keys: a typo here (e.g. 'scaleMn') would silently fall back to
        # the no-perturbation default and quietly run stock objects -> validate loudly.
        oa = env_cfg.get('objectAug') or {}
        bado = sorted(k for k in oa if k not in
                      ('enable', 'scaleMin', 'scaleMax', 'yawRad', 'translateM', 'massExp', 'geom'))
        if bado:
            raise ValueError(f"[intermimic] unknown objectAug key(s) {bado} (only 'enable', "
                             f"'scaleMin', 'scaleMax', 'yawRad', 'translateM', 'massExp', 'geom').")
        # objectAug.geom (anisotropic geometry variants) sub-keys.
        bg = sorted(k for k in (oa.get('geom') or {}) if k not in
                    ('enable', 'numVariants', 'anisoMin', 'anisoMax'))
        if bg:
            raise ValueError(f"[intermimic] unknown objectAug.geom key(s) {bg} (only 'enable', "
                             f"'numVariants', 'anisoMin', 'anisoMax').")

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._validate_env_config(cfg["env"])
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterMimic.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']
        self.more_rigid = cfg['env']['moreRigid']
        self.rollout_length = cfg['env']['rolloutLength']
        self.psi = cfg['env'].get('physicalBufferSize', 1)
        # cpuMotionData: keep the (large) reference-motion tensors on CPU and stream
        # the in-flight frames to GPU per step instead of holding every clip resident
        # in VRAM. Trades a small per-step transfer for ~all the motion data's memory,
        # so the curriculum scales to far more source data than fits on the GPU.
        self._cpu_motion = cfg['env'].get('cpuMotionData', False)
        # Evaluation only works with stateInit "Start"
        state_init_is_start = (state_init == "Start")
        self.enable_evaluation = cfg['env'].get('enableEvaluation', False) and state_init_is_start
        if cfg['env'].get('enableEvaluation', False) and not state_init_is_start:
            print(f"Warning: Evaluation is disabled because stateInit is '{state_init}' (must be 'Start')")
        # --- Motion file discovery ---
        # We enumerate every .pt in self.motion_file (which is actually the
        # *directory* containing motion files at this point — the field is
        # reassigned to a list of paths a few lines below). The dataSub filter
        # selects which motion files load by the subject number embedded in
        # the filename. For identity-pair files (sub<N>_<obj>_<idx>.pt, the
        # only ones currently on disk) src == tgt == N, so this is effectively
        # a source-subject filter. The runtime *target body* the env controls
        # is decoupled from this — it's set per-env by `subjectBodies` in
        # humanoid.py, which selects which MJCF each env loads.
        # _parse_motion_subject also handles a hypothetical sub<src>to<tgt>_*
        # cross-body file format (no such files on disk today), but the body
        # actually simulated still comes from subjectBodies regardless.
        all_files = os.listdir(self.motion_file)
        # `dataSub` in cfg is a list of strings like ['sub2', 'sub8'] — same
        # convention as before. Convert to a set of ints once for matching.
        data_sub_nums = {int(s[3:]) for s in cfg['env']['dataSub']}
        # `dataObjects`: optional cfg list like ['largetable']. If absent or
        # empty, all objects pass through. Used by single-object teachers so
        # one cfg specializes to one HOI task.
        data_objects_cfg = cfg['env'].get('dataObjects', None)
        data_objects = set(data_objects_cfg) if data_objects_cfg else None
        # `maxClipsPerObject`: optional int cap on the number of clips kept
        # per (source, object) bucket after filtering. None = no cap.
        # Used to control for sample count across objects (e.g. sub2 has
        # 17 largetable clips but only 10 woodchair clips — capping both
        # at 10 makes the per-object teachers comparable).
        max_clips_per_object = cfg['env'].get('maxClipsPerObject', None)

        parsed = []  # list of (full_path, source_num, target_num, object_name)
        for fname in all_files:
            try:
                src_num, tgt_num = _parse_motion_subject(fname)
            except ValueError:
                # Not a recognized motion file (e.g. a README.md sitting in
                # the dir). Skip silently rather than crash so test runs
                # tolerate stray files.
                continue
            if tgt_num not in data_sub_nums:
                continue
            obj_name = fname.rsplit('.', 1)[0].split('_')[-2]
            if data_objects is not None and obj_name not in data_objects:
                continue
            parsed.append((os.path.join(self.motion_file, fname), src_num, tgt_num, obj_name))

        # Sort by file path for determinism (mirrors the old behavior of
        # sorted(...) on the list of filenames).
        parsed.sort(key=lambda t: t[0])

        if max_clips_per_object is not None:
            # Group by (source, object) and take first N per group after sort.
            from collections import defaultdict
            counts = defaultdict(int)
            capped = []
            for entry in parsed:
                key = (entry[1], entry[3])  # (source_num, object_name)
                if counts[key] < max_clips_per_object:
                    capped.append(entry)
                    counts[key] += 1
            parsed = capped

        # Fail loudly on an empty / partially-matched dataset. A mistyped
        # dataSub number or wrong motion dir otherwise yields zero (or fewer)
        # clips silently -> training/eval runs on nothing and produces garbage.
        if not parsed:
            raise ValueError(
                f"[intermimic] no motion files matched dataSub={sorted(data_sub_nums)} "
                f"objects={data_objects} under '{self.motion_file}'. Check the "
                f"motion_file dir and dataSub numbers (misspelled?).")
        # Per-subject presence is only a WARNING, not an error: a subject can
        # legitimately have zero clips for a given dataObjects filter (e.g. it
        # never handled that object), and crashing a valid run on that would be
        # a false positive. The empty-total case above is the unambiguous error.
        matched = {p[2] for p in parsed}
        missing = sorted(set(data_sub_nums) - matched)
        if missing:
            print(f"[intermimic] WARNING: dataSub subject(s) {missing} matched ZERO "
                  f"motion files under '{self.motion_file}' (objects={data_objects}). "
                  f"Training/eval will proceed WITHOUT them -- check for a typo if "
                  f"that's unexpected.", flush=True)

        self.motion_file = [p[0] for p in parsed]
        source_subject_nums = [p[1] for p in parsed]
        target_subject_nums = [p[2] for p in parsed]

        # --- Object name parsing ---
        # Unchanged: object name is always the 2nd-to-last underscore-separated
        # token, regardless of whether the file is identity or cross-body.
        self.object_name = [motion_example.split('_')[-2] for motion_example in self.motion_file]
        object_name_set = sorted(list(set(self.object_name)))
        # Construct device string before super().__init__() since self.device is set there
        if device_type == "cuda" or device_type == "GPU":
            self._init_device = "cuda:" + str(device_id)
        else:
            self._init_device = "cpu"
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).to(self._init_device)
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        self.ref_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 + 52 * 3 + 52 + 1
        self.num_motions = len(self.motion_file)

        # --- Cross-body subject + betas tensors ---
        # target_subject_index is the renamed/equivalent of the old
        # `dataset_index`: per-motion subject number, used by env reset code
        # to assign envs to a specific subject's body. We keep `dataset_index`
        # pointing at the target subject for backwards compat with the rest of
        # the file (line ~480 reads `self.dataset_index[self.data_id]`).
        self.source_subject_index = to_torch(source_subject_nums, dtype=torch.long).to(self._init_device)
        self.target_subject_index = to_torch(target_subject_nums, dtype=torch.long).to(self._init_device)
        self.dataset_index = self.target_subject_index  # alias for back-compat

        # Load per-subject betas lookup, then assemble per-motion betas tensors
        # so each motion file has (source_betas, target_betas) ready to go for
        # observation conditioning (task #7). The cfg field is OPTIONAL — if
        # absent, we fall back to zero betas (which means standard InterMimic
        # behavior with no shape conditioning, used by the existing single-
        # subject teacher training).
        betas_file = cfg['env'].get('betas_file', None)
        # Flag for the obs builder. When True, the 32-dim
        # (source_betas, target_betas) vector is appended to obs_buf. When
        # False (existing single-subject configs), nothing is appended and
        # numObs keeps its old value. The user is responsible for setting
        # numObs to base+32 in cfg when they enable betas_file.
        self._use_betas_obs = betas_file is not None
        # Transformer policy obs (opt-in). When True, obs_buf is built as 4
        # multi-horizon tokens (delta_t = 0, 1, 4, 16) instead of the MLP's
        # 2-horizon (1, 16), so the temporal transformer can attend over them.
        # Betas (if used) are folded into EACH token so the net's
        # view(batch, 4, -1) reshape stays clean: numObs = 4 * 1599 = 6396, or
        # 4 * (1599 + 32) = 6524 with betas. Default False => stock MLP obs.
        self._use_transformer_obs = cfg['env'].get('useTransformerObs', False)
        if betas_file is not None:
            from ...utils.path_utils import resolve_repo_path
            betas_path = resolve_repo_path(betas_file)
            betas_lookup = _load_betas_npz(betas_path)
            # Stash for later: env_target_betas (built after super().__init__()
            # when self._env_subject_idx exists from chunk 1) needs to look up
            # betas by env body assignment, not by motion file metadata.
            self._betas_lookup = betas_lookup
            self.source_betas = to_torch(
                np.stack([betas_lookup[n] for n in source_subject_nums], axis=0),
                dtype=torch.float,
            ).to(self._init_device)
            # NOTE: target_betas no longer derived from motion file metadata;
            # built per-env from _env_subject_idx after super().__init__()
            # — see self._env_target_betas below.
        else:
            # No betas file: fill source_betas with zeros for backward compat.
            self._betas_lookup = None
            self.source_betas = torch.zeros((self.num_motions, 16),
                                            dtype=torch.float, device=self._init_device)

        # --- PHYSICS objectAug (training): per-env object scale + per-episode yaw/
        # translate + reward-term toggles. Ported from objectaug-experiment. OFF =>
        # scale=1, gates no-op => byte-identical to stock. NOTE: per-object realistic
        # ranges, object CONDITIONING (obs change), and the curriculum schedule are
        # deliberate FOLLOW-UPS -- this is the core physics-perturbation port only.
        oa = cfg['env'].get('objectAug', None)
        self._object_aug = bool(oa) and bool(oa.get('enable', False))
        _ne = int(cfg['env']['numEnvs'])
        if self._object_aug:
            _slo = float(oa.get('scaleMin', 1.0)); _shi = float(oa.get('scaleMax', 1.0))
            self._oa_yaw = float(oa.get('yawRad', 0.0))
            self._oa_translate = float(oa.get('translateM', 0.0))
            self._oa_mass_exp = float(oa.get('massExp', 2.0))   # mass ~ scale**massExp (2 shell, 3 solid)
            _g = torch.Generator().manual_seed(12345)           # fixed seed: reproducible per-env scale
            self._oa_scale = (torch.rand(_ne, generator=_g) * (_shi - _slo) + _slo).to(self._init_device)
            print(f"[objectAug] ON: scale U({_slo},{_shi}) yaw+/-{self._oa_yaw:.3f}rad "
                  f"translate+/-{self._oa_translate}m massExp={self._oa_mass_exp}", flush=True)
        else:
            self._oa_yaw = 0.0; self._oa_translate = 0.0; self._oa_mass_exp = 3.0
            self._oa_scale = torch.ones(_ne, device=self._init_device)

        # --- GEOMETRY augmentation (anisotropic): per-env non-uniform object
        # proportions for shape robustness. Isaac Gym's set_actor_scale is uniform-
        # only and collision geometry is baked at prepare_sim, so distinct shapes
        # need distinct ASSETS: each object category gets `numVariants` pre-baked
        # variant URDFs with a per-axis scale (built in _load_target_asset), and each
        # env is pinned to one (object, variant). Composes with the uniform objectAug
        # scale above. Requires objectAug on. OFF => 1 variant, aniso=1 => identical
        # to the objectAug-only path. The stock object-match terms don't survive a
        # geometry change, so pair this with objectTermsEnable:false + rewardTerms.hold.
        _geom = (oa.get('geom', {}) or {}) if self._object_aug else {}
        self._geom_aug = bool(_geom.get('enable', False))
        _nobj = len(self.object_name)
        if self._geom_aug:
            self._geom_nvar = max(1, int(_geom.get('numVariants', 8)))
            _amin = float(_geom.get('anisoMin', 0.80)); _amax = float(_geom.get('anisoMax', 1.20))
            _gg = torch.Generator().manual_seed(6789)   # fixed seed: reproducible variants
            # per (object, variant) anisotropic scale triple (sx,sy,sz). Variant 0 is
            # the UNPERTURBED shape (all ones) so each object's true geometry stays in
            # the mix and the policy always sees some nominal shapes.
            self._geom_aniso = (torch.rand(_nobj, self._geom_nvar, 3, generator=_gg)
                                * (_amax - _amin) + _amin)
            if self._geom_nvar >= 1:
                self._geom_aniso[:, 0, :] = 1.0
            # per-env variant index (seeded), then per-env aniso = aniso[obj, variant].
            self._env_variant = torch.randint(0, self._geom_nvar, (_ne,), generator=_gg)
            _objidx = torch.arange(_ne) % _nobj
            self._geom_aniso_per_env = self._geom_aniso[_objidx, self._env_variant].to(self._init_device)
            print(f"[objectAug.geom] ON: {self._geom_nvar} variants/object, aniso "
                  f"U({_amin},{_amax}) per axis, {_nobj} objects => "
                  f"{_nobj * self._geom_nvar} assets to bake", flush=True)
        else:
            self._geom_nvar = 1
            self._env_variant = torch.zeros(_ne, dtype=torch.long)
            self._geom_aniso_per_env = torch.ones(_ne, 3, device=self._init_device)
        # Combined per-env object-point scale = uniform objectAug scale * anisotropic
        # geom. Applied wherever the surface point cloud is transformed (policy obs +
        # hold reward + obj reward). When both off it's all-ones (a no-op).
        self._obj_pts_scale = (self._oa_scale.view(-1, 1, 1)
                               * self._geom_aniso_per_env.view(-1, 1, 3)).to(self._init_device)

        # Reward-term toggles. hold ported from objectaug-experiment; pose already exists.
        _rt = cfg['env'].get('rewardTerms', {}) or {}
        _hold = _rt.get('hold', {}) or {}
        self._hold_term_enable = bool(_hold.get('enable', False))
        self._hold_lambda = float(_hold.get('lambda', 5.0))
        # Stock object-match terms (ro*rig*rcg). ON by default => stock reward. Toggle OFF
        # for objectAug runs where the perturbed object makes them unachievable.
        self._object_terms_enable = bool(cfg['env'].get('objectTermsEnable', True))
        # Opt-in mass-hold verification print (OBJECTAUG_DEBUG=1): prints per-object
        # total_mass vs aug for the first envs so you can confirm mass tracks
        # scale**massExp (not scale**3) before trusting a run. No training effect.
        self._oa_debug = os.environ.get('OBJECTAUG_DEBUG') == '1'

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)

        # Body-dependent features REQUIRE per-subject bodies. If subjectBodies is
        # missing/misspelled, self.subject_bodies is None and these silently no-op
        # (canonical betas, no body-norm, no pair weights) -- fail loudly instead.
        if getattr(self, 'subject_bodies', None) is None:
            for _k in ('betas_file', 'bodyNormalizedReward',
                       'subjectPairWeightsFile', 'subjectHeightsFile'):
                if cfg['env'].get(_k):
                    raise ValueError(
                        f"[intermimic] '{_k}' is configured but 'subjectBodies' is "
                        f"absent/empty (misspelled?). This feature needs per-subject "
                        f"bodies; refusing to silently run single-body and ignore it.")

        # ---- TERM_REASON=1: why do episodes END, broken down per body? ----------
        # Success here == "survived to the end of the clip", so a low success rate
        # is ALWAYS some episode ending early -- but four different things can end
        # it (fell below terminationHeight / NaN obs / interaction divergence /
        # contact divergence) and nothing distinguished them. A body that falls
        # over and a body that drops the object need opposite fixes.
        # Diagnostic only: counts are accumulated in compute_hoi_reset and printed
        # periodically. Never touches reward, reset, or the policy.
        self._term_reason = os.environ.get('TERM_REASON', '0') == '1'
        self._term_reason_every = int(os.environ.get('TERM_REASON_EVERY', '2000'))
        if self._term_reason:
            self._term_labels = ['completed', 'fell', 'nan_obs', 'ig_diverge', 'contact_diverge']
            n_bodies = len(self.subject_bodies) if getattr(self, 'subject_bodies', None) else 1
            self._term_counts = torch.zeros((n_bodies, len(self._term_labels)),
                                            dtype=torch.long, device=self.device)
            self._term_episodes = torch.zeros(n_bodies, dtype=torch.long, device=self.device)
            self._term_steps = 0
            print(f"[term_reason] enabled -- reporting every {self._term_reason_every} steps "
                  f"across {n_bodies} body(ies)")

        # Per-env target_betas always reflects the body in sim (chunk 1's
        # _env_subject_idx). When subjectBodies is set, look up each body's
        # betas from the npz; when absent, all envs run canonical (β=0).
        self._env_target_betas = None
        if self._use_betas_obs:
            if getattr(self, 'subject_bodies', None) is not None:
                body_subject_nums = [int(s[3:]) for s in self.subject_bodies]
                body_betas = to_torch(
                    np.stack([self._betas_lookup[n] for n in body_subject_nums], axis=0),
                    dtype=torch.float,
                ).to(self.device)
                self._env_target_betas = body_betas[self._env_subject_idx]
                print(f"[intermimic] built _env_target_betas {tuple(self._env_target_betas.shape)} from subject_bodies={self.subject_bodies}", flush=True)
            else:
                self._env_target_betas = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)
                print(f"[intermimic] built _env_target_betas (canonical β=0) for {self.num_envs} envs", flush=True)

        # Body-size-normalized reward (feature flag). When enabled, pose-error
        # terms in compute_humanoid_reward get divided by per-env body height
        # to remove the size bias (default reward penalizes smaller bodies more
        # for the same physical tracking deviation).
        self._body_normalized_reward = cfg['env'].get('bodyNormalizedReward', False)
        self._env_body_height = None
        if self._body_normalized_reward:
            if getattr(self, 'subject_bodies', None) is not None:
                # An optional heights file EXTENDS the hardcoded SUBJECT_HEIGHTS --
                # needed for synthetic bodies (sub100+) that aren't in the dict.
                heights_map = dict(SUBJECT_HEIGHTS)
                hfile = cfg['env'].get('subjectHeightsFile', None)
                if hfile is not None:
                    import json
                    heights_map.update({int(k): float(v)
                                        for k, v in json.load(open(hfile)).items()})
                heights = [heights_map[int(s[3:])] for s in self.subject_bodies]
            else:
                heights = [1.585]   # canonical SMPL-X height
            body_heights_per_subj = to_torch(heights, dtype=torch.float).to(self.device)
            self._env_body_height = body_heights_per_subj[self._env_subject_idx]
            print(f"[intermimic] body-normalized reward enabled; per-env body heights {tuple(self._env_body_height.shape)} (min {self._env_body_height.min().item():.3f}, max {self._env_body_height.max().item():.3f})", flush=True)

        # --- Term 1: parent-relative joint-angle pose reward (opt-in factor) ---
        # Ported from the objectaug-experiment branch. Compares simulated vs
        # reference dof_pos (the 51x3=153 parent-relative joint DOFs) and folds
        # exp(-lambda * sum_j (dof_ref - dof_sim)^2) into the reward PRODUCT when
        # rewardTerms.pose.enable is set. Default OFF => reward byte-identical to
        # stock. The error is a SUM over 153 DOFs, so lambda ~0.02 is roughly the
        # rotation term's 2.5 expressed per-DOF. See _compute_pose_reward.
        pose_cfg = (cfg['env'].get('rewardTerms', {}) or {}).get('pose', {}) or {}
        self._pose_term_enable = bool(pose_cfg.get('enable', False))
        self._pose_lambda = float(pose_cfg.get('lambda', 0.02))
        # Env-var-gated dof-alignment sanity print (no effect on training).
        self._pose_reward_debug = os.environ.get('POSE_REWARD_DEBUG') == '1'
        if self._pose_term_enable:
            print(f"[intermimic] pose reward (relative joint-angle) enabled; "
                  f"lambda={self._pose_lambda}", flush=True)

        # --- Per-(body, source) pair sampling weights (curriculum balancing) ---
        # Optional. cfg 'subjectPairWeightsFile' points at a JSON mapping
        # "b{B}_s{S}" -> float weight (B, S = subject numbers). When set, motion
        # sampling at reset is weighted so each env (whose body is fixed at
        # creation) picks a source S in proportion to W[body, S]. This lets the
        # curriculum controller equalize exposure per (body, source) PAIR rather
        # than just per subject. When absent, sampling is uniform — identical to
        # the original behavior, so all existing configs are unaffected.
        # Precomputed as a (num_bodies, num_motions) tensor: row = env's body
        # index (self._env_subject_idx), col = motion, value = W[body, motion's
        # source]. A 0-sum row over an object's valid motions falls back to
        # uniform at sample time (e.g. a pair with no shared object).
        self._pair_weight_per_body = None
        pair_weights_file = cfg['env'].get('subjectPairWeightsFile', None)
        if pair_weights_file is not None and getattr(self, 'subject_bodies', None):
            import json
            from ...utils.path_utils import resolve_repo_path
            with open(resolve_repo_path(pair_weights_file)) as f:
                pair_w = json.load(f)
            body_subject_nums = [int(s[3:]) for s in self.subject_bodies]
            src_nums = self.source_subject_index.tolist()
            W = torch.ones((len(body_subject_nums), self.num_motions),
                           dtype=torch.float, device=self.device)
            for bi, b in enumerate(body_subject_nums):
                for mi, s in enumerate(src_nums):
                    _key = f"b{b}_s{s}"
                    if _key not in pair_w:
                        raise KeyError(
                            f"[intermimic] pair weight '{_key}' missing from "
                            f"{pair_weights_file}. curriculum_runner writes 0.0 for "
                            f"masked pairs, so a missing key means the weights file is "
                            f"incomplete/mismatched -- refusing to default to full "
                            f"sampling (1.0), which would leak a held-out pair.")
                    W[bi, mi] = float(pair_w[_key])
            self._pair_weight_per_body = W
            # Keep the number maps around for the realized-sample counter below.
            self._body_subject_nums = body_subject_nums      # body row idx -> subject num
            self._src_nums = src_nums                        # motion col idx -> source num
            # Report any (body, source) pair that is physically unreachable:
            # body bi never co-occurs (in env creation) with an object that
            # source s performed, so no env can ever realize that pair.
            unreachable = []
            for bi, b in enumerate(body_subject_nums):
                # objects body bi's envs physically own: bi covers object o iff
                # some env index e has e % num_bodies == bi and e % num_obj == o.
                body_objs = {int(e % len(self.object_name))
                             for e in range(self.num_envs)
                             if int(e % len(body_subject_nums)) == bi}
                for s in sorted(set(src_nums)):
                    s_motions = [mi for mi, ss in enumerate(src_nums) if ss == s]
                    s_objs = {int(self.object_id[mi]) for mi in s_motions}
                    if not (body_objs & s_objs):
                        unreachable.append((b, s))
            print(f"[intermimic] loaded pair sample weights from {pair_weights_file}: "
                  f"W {tuple(W.shape)} bodies={body_subject_nums}", flush=True)
            if unreachable:
                print(f"[intermimic] WARNING: {len(unreachable)} (body,source) pairs "
                      f"are physically unreachable (no shared object): {unreachable}", flush=True)

        # --- Realized (source, body) sample counter (curriculum measured exposure) ---
        # Optional. cfg 'pairSampleCountsFile' is an OUTPUT path. When set (and
        # pair weights are loaded), we tally how many times each (source, body)
        # pair is actually sampled at reset and periodically flush the tally to
        # that file. The curriculum controller reads it to (a) reweight the next
        # sub-stage from MEASURED exposure rather than an estimate, and (b) detect
        # leaks — a pair we masked (weight 0) but which still has a nonzero count
        # was sampled via the uniform fallback for an env with no live pair.
        self._pair_sample_counts = None
        counts_file = cfg['env'].get('pairSampleCountsFile', None)
        if counts_file is not None and self._pair_weight_per_body is not None:
            from ...utils.path_utils import resolve_repo_path
            self._pair_counts_path = str(resolve_repo_path(counts_file, must_exist=False))
            self._pair_sample_counts = {}
            self._pair_counts_flush_every = 200   # flush every N reset batches
            self._pair_counts_since_flush = 0
            print(f"[intermimic] realized pair-sample counting on -> {counts_file}", flush=True)

        # --- Dead-env gradient mask (curriculum hard guarantee) ---
        # Optional. cfg 'maskDeadEnvs' True => publish a per-env boolean mask so
        # the agent can zero the PPO gradient from "dead" envs — those whose
        # fixed (body, object) has NO live pair this sub-stage (so their reset
        # falls back to a not-yet-scheduled pair and would otherwise leak it into
        # training). Body and object are fixed per env at creation, so this mask
        # is STATIC for the whole sub-stage: compute it once and hand it to the
        # agent via self.extras (same channel as 'terminate'). When off (default)
        # the key is absent and the agent's masking is a no-op, so every other
        # training run is unaffected.
        self._live_env_mask = None
        if cfg['env'].get('maskDeadEnvs', False) and self._pair_weight_per_body is not None:
            no = len(self.object_name)
            nb = len(self._body_subject_nums)
            # live_bo[b, o] True iff body b's envs (which hold object o) have at
            # least one positively-weighted pair among object o's motions.
            live_bo = torch.zeros((nb, no), dtype=torch.bool, device=self.device)
            for o in range(no):
                valid = torch.where(self.obj2motion[o] == 1)[0]
                if valid.numel() == 0:
                    continue
                live_bo[:, o] = self._pair_weight_per_body[:, valid].sum(dim=1) > 0
            obj_per_env = torch.arange(self.num_envs, device=self.device) % no
            self._live_env_mask = live_bo[self._env_subject_idx, obj_per_env].float()
            self.extras["live_env_mask"] = self._live_env_mask
            n_dead = int((self._live_env_mask == 0).sum())
            print(f"[intermimic] dead-env gradient masking ON: {n_dead}/{self.num_envs} "
                  f"envs have no live pair this sub-stage (gradients masked).", flush=True)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._curr_reward = torch.zeros([self.num_envs, cfg['env']['rolloutLength']], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, cfg['env']['rolloutLength'], 332], device=self.device, dtype=torch.float)
        self._build_target_tensors()
        # --- video recording setup ---
        # If RECORD_VIDEO env var is set, the rl_games player handles recording
        # via its own camera sensor (intermimic_players.py). Don't ALSO create
        # an env-side camera — two cameras in play_dataset_step caused segfaults.
        _record_video_external = os.environ.get("RECORD_VIDEO") is not None
        if self.play_dataset and not _record_video_external:
            cam_props = gymapi.CameraProperties()
            cam_props.width, cam_props.height = 1280, 720
            self._video_cam = self.gym.create_camera_sensor(self.envs[0], cam_props)
            self.gym.set_camera_location(
                self._video_cam, self.envs[0],
                gymapi.Vec3(3.0, 3.0, 2.0),
                gymapi.Vec3(0.0, 0.0, 1.0),
            )
            import imageio
            os.makedirs('replay_frames', exist_ok=True); self._video_writer = None; self._video_frame_idx = 0
            self._video_height, self._video_width = 720, 1280
            # END OF modified part

        return

    def post_physics_step(self):
        super().post_physics_step()
        # GPU memory diagnostic (read-only, throttled): watch total usage climb
        # toward the OOM point. torch alloc = our tensors; GPU used = incl PhysX.
        self._memchk_n = getattr(self, '_memchk_n', 0) + 1
        if self._memchk_n % 200 == 1:
            free, total = torch.cuda.mem_get_info()
            print(f"[mem] step {self._memchk_n}: torch {torch.cuda.memory_allocated() / 1024 ** 3:.2f}G | "
                  f"GPU used {(total - free) / 1024 ** 3:.1f}/{total / 1024 ** 3:.0f}G", flush=True)
        return

    def debug_env_tags(self, env_ids):
        """Best-effort '<body>/<object>×count' summary for a set of env ids, so
        the agent's blow-up guards can say WHICH (body, object) combos exploded
        instead of just a count. Static per-env identity: object = env%num_objects,
        body = subject behind _env_subject_idx. Defensive by design -- a logging
        helper must NEVER raise into the training hot path, so any failure just
        yields a marker string."""
        try:
            from collections import Counter
            if torch.is_tensor(env_ids):
                env_ids = env_ids.detach().flatten().cpu().tolist()
            no = len(self.object_name)
            tags = Counter()
            for e in env_ids:
                e = int(e)
                obj = self.object_name[e % no] if no else '?'
                if hasattr(self, '_body_subject_nums') and hasattr(self, '_env_subject_idx'):
                    body = f"sub{int(self._body_subject_nums[int(self._env_subject_idx[e])])}"
                else:
                    body = 'sub?'
                tags[f"{body}/{obj}"] += 1
            return ', '.join(f"{k}×{v}" for k, v in tags.most_common(8)) or '(none)'
        except Exception as ex:  # noqa: BLE001 -- must not crash training
            return f"<tag-failed: {ex}>"

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):

        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        # Process data on CPU first, then move to GPU at the end
        object_points_cpu = self.object_points.cpu()
        object_id_cpu = self.object_id.cpu()

        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path)[startk:]
            loaded_dict['hoi_data'] = hoi_data.detach()  # Keep on CPU for processing


            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

            loaded_dict['dof_vel'] = []

            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])),loaded_dict['dof_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone()
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])),loaded_dict['body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()

            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])),loaded_dict['obj_pos_vel']),dim=0)


            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])),loaded_dict['obj_rot_vel']),dim=0)

            # Use CPU tensors for object points computation
            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, object_points_cpu[object_id_cpu[idx]].shape[0], 1).view(-1, 4)
            object_points_extend = object_points_cpu[object_id_cpu[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], object_points_cpu[object_id_cpu[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)

            ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],52,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)
            loaded_dict['ig'] = ref_ig
            loaded_dict['contact_obj'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
            loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())
            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone()

            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3)
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])),loaded_dict['body_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), 
                                                    loaded_dict['root_rot'].clone(), 
                                                    loaded_dict['dof_pos'].clone(), 
                                                    loaded_dict['dof_vel'].clone(),
                                                    loaded_dict['body_pos'].clone(),
                                                    loaded_dict['body_rot'].clone(),
                                                    loaded_dict['body_pos_vel'].clone(),
                                                    loaded_dict['body_rot_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(), 
                                                    loaded_dict['obj_rot_vel'].clone(),
                                                    loaded_dict['ig'].clone(),
                                                    loaded_dict['contact_human'].clone(),
                                                    loaded_dict['contact_obj'].clone(),
                                                    ),dim=-1)
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), 
                                loaded_dict['root_rot'].clone(), 
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(), 
                                loaded_dict['dof_pos'].clone(), 
                                loaded_dict['dof_vel'].clone(), 
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(max_episode_length) + initk
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long, device=self.device) + initk
        hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        # Stack on CPU. With cpuMotionData we KEEP them on CPU and stream per step
        # (_motion_gather); otherwise move the whole thing to GPU as before.
        hoi_data = torch.stack(hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, topk, 1, 1)
        if not self._cpu_motion:
            hoi_data = hoi_data.to(self.device)
            self.hoi_refs = self.hoi_refs.to(self.device)

        # --- GPU memory diagnostic (read-only) -- how big are the motion tensors,
        # and total GPU used (incl PhysX, via mem_get_info) right after loading them?
        _gb = lambda t: t.element_size() * t.nelement() / 1024 ** 3
        free, total = torch.cuda.mem_get_info()
        print(f"[mem] motion tensors: hoi_data {_gb(hoi_data):.2f}G {tuple(hoi_data.shape)} + "
              f"hoi_refs {_gb(self.hoi_refs):.2f}G {tuple(self.hoi_refs.shape)} = "
              f"{_gb(hoi_data) + _gb(self.hoi_refs):.2f}G "
              f"{'on CPU (streamed per step)' if self._cpu_motion else 'on GPU'}", flush=True)
        print(f"[mem] after motion load: torch {torch.cuda.memory_allocated() / 1024 ** 3:.2f}G | "
              f"GPU used {(total - free) / 1024 ** 3:.1f}/{total / 1024 ** 3:.0f}G (incl PhysX/other)",
              flush=True)

        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2]), device=self.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)

        # Evaluation metrics tracking per sequence (only if evaluation is enabled)
        if self.enable_evaluation:
            self._max_execution_steps = torch.zeros([self.num_motions], device=self.device, dtype=torch.long)
            self._human_pose_error_per_seq_step = torch.ones([self.num_motions, max_length], device=self.device, dtype=torch.float) * 1e6
            self._object_pose_error_per_seq_step = torch.ones([self.num_motions, max_length], device=self.device, dtype=torch.float) * 1e6
            self._best_human_pose_error_per_seq = torch.ones([self.num_motions], device=self.device, dtype=torch.float) * 1e6
            self._best_object_pose_error_per_seq = torch.ones([self.num_motions], device=self.device, dtype=torch.float) * 1e6
            # Track visit counts for balanced sampling
            self._sequence_visit_count = torch.zeros([self.num_motions], device=self.device, dtype=torch.long)

        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)
        return hoi_data

    def create_component_stat(self, loaded_dict):
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]

        # Precompute the sizes for each component.
        data_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.data_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.data_component_index = [sum(data_component_sizes[:i]) for i in range(len(data_component_sizes) + 1)]

        self.ref_component_order = [
            'root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_vel', 'obj_pos', 'obj_rot', 
            'obj_pos_vel', 'obj_rot_vel'
        ]

        # Precompute the sizes for each component.
        ref_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.ref_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.ref_component_index = [sum(ref_component_sizes[:i]) for i in range(len(ref_component_sizes) + 1)]

    def _motion_gather(self, tensor, idx):
        """Index a reference-motion tensor (hoi_data / hoi_refs) that may live on CPU
        under cpuMotionData. `idx` is the full index tuple (advanced-index tensors +
        an optional trailing `slice`). Returns the gathered slice on self.device.
        When the tensor is already on GPU this is plain indexing -- identical result,
        zero overhead. Correct by construction: t[(a,b,slice(s,e))] == t[a,b,s:e], and
        moving the index to CPU / the small result to GPU changes neither which nor
        what values are selected (verified)."""
        if tensor.is_cuda:
            return tensor[idx]
        idx = tuple(i.cpu() if torch.is_tensor(i) else i for i in idx)
        return tensor[idx].to(self.device, non_blocking=True)

    def extract_ref_component(self, var_name, data_id, ref_index, t):
        index = self.ref_component_order.index(var_name)

        # The number of columns to extract for this component.
        start = self.ref_component_index[index]
        end = self.ref_component_index[index+1]

        return self._motion_gather(self.hoi_refs, (data_id, ref_index, t, slice(start, end)))


    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        index = self.data_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]
        
        if ref and data_id is not None and t is not None:
            return self._motion_gather(self.hoi_data, (data_id, t, slice(start, end)))
        
        if obs is not None:
            return obs[..., start:end]

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _write_geom_variant_urdf(self, asset_root, object_name, k, scale_xyz):
        """Write a geometry-variant URDF for object k with an anisotropic mesh scale.

        The base URDFs all carry `scale="1.0 1.0 1.0"` on their visual + collision
        <mesh> elements; we clone the URDF verbatim and swap that for the per-axis
        triple, pointing at the SAME .obj (so the big mesh is never duplicated and the
        relative mesh path still resolves against asset_root). VHACD re-decomposes the
        scaled shape at load. Returns the variant URDF's path relative to asset_root.
        """
        sx, sy, sz = scale_xyz
        with open(os.path.join(asset_root, object_name + ".urdf")) as f:
            txt = f.read()
        new = txt.replace('scale="1.0 1.0 1.0"', f'scale="{sx:.6f} {sy:.6f} {sz:.6f}"')
        # deterministic name; regenerated each run, so gitignored (assets/objects/.gitignore).
        rel = f"{object_name}.geomv{k}.urdf"
        with open(os.path.join(asset_root, rel), 'w') as f:
            f.write(new)
        return rel

    def _load_target_asset(self): # smplx
        asset_root = resolve_data_path("assets", "objects")
        self._target_asset = []
        points_num = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            obj_file = resolve_data_path("assets", "objects", "objects", object_name, object_name + ".obj")
            max_convex_hulls = 64
            density = self.object_density

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000


            if self._geom_aug:
                # bake one asset per geometry variant (anisotropic-scaled URDF, same
                # .obj). Nested: self._target_asset[obj] = [v0_asset, v1_asset, ...].
                variant_assets = []
                for k in range(self._geom_nvar):
                    sx, sy, sz = self._geom_aniso[i, k].tolist()
                    rel = self._write_geom_variant_urdf(str(asset_root), object_name, k, (sx, sy, sz))
                    variant_assets.append(self.gym.load_asset(self.sim, str(asset_root), rel, asset_options))
                self._target_asset.append(variant_assets)
                if self._oa_debug:
                    print(f"[geomchk] {object_name}: baked {self._geom_nvar} variant assets "
                          f"(v0=identity, aniso e.g. {self._geom_aniso[i, min(1, self._geom_nvar-1)].tolist()})",
                          flush=True)
            else:
                self._target_asset.append(self.gym.load_asset(self.sim, str(asset_root), asset_file, asset_options))

            mesh_obj = trimesh.load(str(obj_file), force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

        self.object_points = torch.stack(self.object_points, dim=0).to(self.device)
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        obj_idx = env_id % len(self.object_name)
        if self._geom_aug:
            # this env's pre-baked geometry variant (anisotropic-scaled asset).
            obj_asset = self._target_asset[obj_idx][int(self._env_variant[env_id])]
        else:
            obj_asset = self._target_asset[obj_idx]
        target_handle = self.gym.create_actor(env_ptr, obj_asset, default_pose, self.object_name[obj_idx], col_group, col_filter, segmentation_id)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.05
            props[p_idx].friction = 0.6
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.01
            if self.object_name[env_id % len(self.object_name)] == 'plasticbox' or self.object_name[env_id % len(self.object_name)] == 'trashcan':
                props[p_idx].rest_offset = 0.015
            else:
                props[p_idx].rest_offset = 0.002
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        self._target_handles.append(target_handle)
        aug = float(self._oa_scale[env_id])                      # 1.0 when objectAug off
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size * aug)
        # Mass tracking (uniform scale AND anisotropic geometry variant, unified). The
        # variant URDF already scales collision volume by sx*sy*sz (mass grows by that
        # at fixed density) and set_actor_scale adds a uniform aug**3. So the raw mass
        # factor vs the base solid object is V = (sx*sy*sz)*aug**3. We want mass ~
        # V**(massExp/3) (massExp=3 solid => leave as-is; 2 => shell-ish, mass ~ area),
        # so multiply mass+inertia by V**(massExp/3 - 1). Reduces to the old uniform
        # formula aug**(massExp-3) when the geom aniso is (1,1,1).
        if self._object_aug:
            ax, ay, az = self._geom_aniso_per_env[env_id].tolist()   # (1,1,1) when geom off
            V = (ax * ay * az) * (aug ** 3)
            corr = V ** (self._oa_mass_exp / 3.0 - 1.0)
            if abs(corr - 1.0) > 1e-9:
                props = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
                for bp in props:
                    bp.mass *= corr
                    bp.inertia.x.x *= corr; bp.inertia.x.y *= corr; bp.inertia.x.z *= corr
                    bp.inertia.y.x *= corr; bp.inertia.y.y *= corr; bp.inertia.y.z *= corr
                    bp.inertia.z.x *= corr; bp.inertia.z.y *= corr; bp.inertia.z.z *= corr
                self.gym.set_actor_rigid_body_properties(env_ptr, target_handle, props,
                                                         recomputeInertia=False)

        if self._oa_debug and env_id < 8:
            dbg = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
            ax, ay, az = self._geom_aniso_per_env[env_id].tolist()
            print(f"[masschk] env{env_id} obj={self.object_name[env_id % len(self.object_name)]} "
                  f"aug={aug:.3f} aniso=({ax:.2f},{ay:.2f},{az:.2f}) "
                  f"total_mass={sum(p.mass for p in dbg):.4f} "
                  f"(expect ~ nominal * ((sx*sy*sz)*aug^3)^{self._oa_mass_exp / 3.0:.2f})", flush=True)

        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        return
    
    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 7:10] = self.extract_ref_component('obj_pos_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 10:13] = self.extract_ref_component('obj_rot_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        # objectAug: perturb the INITIAL object pose per episode (scale is baked at env
        # creation). Random yaw about Z + random XY translate for training diversity.
        if self._object_aug and (self._oa_yaw > 0.0 or self._oa_translate > 0.0):
            n = env_ids.shape[0] if torch.is_tensor(env_ids) else len(env_ids)
            if self._oa_yaw > 0.0:
                yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self._oa_yaw
                half = yaw * 0.5
                zq = torch.zeros(n, 4, device=self.device)
                zq[:, 2] = torch.sin(half); zq[:, 3] = torch.cos(half)
                self._target_states[env_ids, 3:7] = quat_mul(zq, self._target_states[env_ids, 3:7])
            if self._oa_translate > 0.0:
                dxy = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self._oa_translate
                self._target_states[env_ids, 0:2] += dxy
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == InterMimic.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == InterMimic.StateInit.Start
              or self._state_init == InterMimic.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == InterMimic.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        self._reset_target(env_ids)

        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _sample_motion_ids(self, env_ids):
        # Pick a motion for each env, restricted to that env's fixed object
        # bucket (env e physically owns object e % num_objects). Uniform by
        # default; if per-(body, source) pair weights are loaded, sample
        # proportional to W[env's body, candidate motion's source].
        n_obj = len(self.object_name)
        if self._pair_weight_per_body is None:
            return to_torch(
                [torch.where(self.obj2motion[e % n_obj] == 1)[0][
                    torch.randint(int(self.obj2motion[e % n_obj].sum()), ())]
                 for e in env_ids],
                device=self.device, dtype=torch.long)
        out = []
        rec = self._pair_sample_counts is not None
        for e in env_ids:
            valid = torch.where(self.obj2motion[e % n_obj] == 1)[0]
            w = self._pair_weight_per_body[self._env_subject_idx[e], valid]
            if float(w.sum()) <= 0.0:
                sel = valid[torch.randint(int(valid.shape[0]), ())]  # no live pair: fallback
            else:
                sel = valid[torch.multinomial(w, 1).squeeze(0)]
            out.append(sel)
            if rec:
                # Tally the realized (source, body) pair. Counting both branches
                # means the fallback's leaks show up against their 0 weight.
                b = self._body_subject_nums[int(self._env_subject_idx[e])]
                s = self._src_nums[int(sel)]
                key = f"b{b}_s{s}"
                self._pair_sample_counts[key] = self._pair_sample_counts.get(key, 0) + 1
        if rec:
            self._maybe_flush_pair_counts()
        return to_torch(out, device=self.device, dtype=torch.long)

    def _maybe_flush_pair_counts(self):
        """Write the realized-sample tally to disk every _pair_counts_flush_every
        reset batches (cheap, atomic via tmp+rename so the controller never reads
        a half-written file)."""
        self._pair_counts_since_flush += 1
        if self._pair_counts_since_flush < self._pair_counts_flush_every:
            return
        self._pair_counts_since_flush = 0
        import json
        import os
        tmp = self._pair_counts_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._pair_sample_counts, f, sort_keys=True)
        os.replace(tmp, self._pair_counts_path)

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        # During evaluation, prioritize undersampled sequences for balanced coverage
        if self.enable_evaluation:
            i = []
            for env_idx in env_ids:
                # Get valid motion indices for this object type
                obj_type = env_idx % len(self.object_name)
                valid_motions = torch.where(self.obj2motion[obj_type] == 1)[0]

                # Get visit counts for valid motions
                visit_counts = self._sequence_visit_count[valid_motions]

                # Sample with inverse probability (prioritize less visited sequences)
                # Add 1 to avoid division by zero
                inv_counts = 1.0 / (visit_counts.float() + 1.0)
                probs = inv_counts / inv_counts.sum()

                # Sample based on inverse visit counts
                sampled_idx = torch.multinomial(probs, 1).item()
                selected_motion = valid_motions[sampled_idx]
                i.append(selected_motion)

            i = to_torch(i, device=self.device, dtype=torch.long)

            # Update visit counts
            for motion_id in i:
                self._sequence_visit_count[motion_id] += 1
        else:
            # Random sampling for training (uniform, or pair-weighted if a
            # subjectPairWeightsFile is configured).
            i = self._sample_motion_ids(env_ids)

        if (self._state_init == InterMimic.StateInit.Random
            or self._state_init == InterMimic.StateInit.Hybrid):
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        elif (self._state_init == InterMimic.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)#.int()

        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

        return

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        ref_reward_sum = 1 / (rewards.sum(dim=0)) 
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = self._sample_motion_ids(env_ids)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = self.compute_obj_observations(root_states, tar_states, ref_obs)
        return obs

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
        # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                                heading_rot_expand_no_hand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                                heading_inv_rot_expand_no_hand.shape[2])
        
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
        
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
    
        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs
    
    def compute_obj_observations(self, root_states, tar_states, ref_obs):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = self.extract_data_component('obj_pos', obs=ref_obs)
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = self.extract_data_component('obj_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = self.extract_data_component('obj_pos_vel', obs=ref_obs)
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = self.extract_data_component('obj_rot_vel', obs=ref_obs)
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
        return obs
    
    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone() 
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
        ref_obs = self._motion_gather(hoi_data, (self.data_id[env_ids], next_ts)).clone()
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs)
        obs = torch.cat([obs, task_obs], dim=-1)    
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs)
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
        
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
        
    def _compute_observations(self, env_ids=None):
        # Horizons stacked into obs_buf: MLP uses 2 (delta_t 1, 16); the
        # transformer policy uses 4 (0, 1, 4, 16) so it can attend over them.
        horizons = [0, 1, 4, 16] if self._use_transformer_obs else [1, 16]
        if (env_ids is None):
            self._curr_ref_obs[:] = self._motion_gather(self.hoi_data, (self.data_id[env_ids], self.progress_buf[env_ids])).clone()
            # (source_betas, target_betas) — 32 dims. source_betas[data_id]: from
            # the motion file; _env_target_betas: the env's actual body in sim.
            betas = torch.cat([self.source_betas[self.data_id], self._env_target_betas], dim=-1) \
                if self._use_betas_obs else None
            self.obs_buf[:] = self._stack_obs_horizons(None, horizons, betas)
        else:
            self._curr_ref_obs[env_ids] = self._motion_gather(self.hoi_data, (self.data_id[env_ids], self.progress_buf[env_ids])).clone()
            betas = torch.cat([self.source_betas[self.data_id[env_ids]],
                               self._env_target_betas[env_ids]], dim=-1) \
                if self._use_betas_obs else None
            self.obs_buf[env_ids] = self._stack_obs_horizons(env_ids, horizons, betas)

    def _stack_obs_horizons(self, env_ids, horizons, betas):
        """Build the stacked policy obs over `horizons` (delta_t values).

        MLP (2 horizons 1,16): [obs@1, obs@16, betas?] -- betas appended ONCE,
        byte-identical to the original teacher obs.
        Transformer (4 horizons 0,1,4,16): each horizon is one token with betas
        folded IN, so the net's view(batch, 4, -1) recovers clean per-token
        features (4 * (1599+32) = 6524, or 4*1599 = 6396 without betas).
        """
        if self._use_transformer_obs:
            toks = []
            for dt in horizons:
                o = self._compute_observations_iter(self.hoi_data, env_ids, dt)
                if betas is not None:
                    o = torch.cat([o, betas], dim=-1)   # fold betas into each token
                toks.append(o)
            return torch.cat(toks, dim=-1)
        obs_terms = [self._compute_observations_iter(self.hoi_data, env_ids, dt) for dt in horizons]
        if betas is not None:
            obs_terms.append(betas)
        return torch.cat(obs_terms, dim=-1)

        return
    
    def _compute_hoi_observations(self, env_ids=None):
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces,
                                                        self._contact_forces,
                                                        self.object_points[self.object_id[self.data_id]] * self._obj_pts_scale,
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel
                                                        )
        return

    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                            local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)    
        
        obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel, 
                         body_pos.reshape(body_pos.shape[0],-1), body_rot.reshape(body_rot.shape[0],-1), body_vel.reshape(body_vel.shape[0],-1), body_rot_vel.reshape(body_rot_vel.shape[0],-1),
                         target_states, ig, contact, target_contact), dim=-1)
        return obs
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = self.compute_hoi_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                                           self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                                           self._enable_early_termination, self._termination_heights, self.start_times,
                                                                           self.rollout_length, self.kinematic_reset, torch.any(self.contact_reset > 10, dim=-1)
                                                                          )

        # Evaluation metrics update (assumes stateInit is "Start", so start_times is 0)
        if self.enable_evaluation:
            reset_id = torch.where(self.reset_buf)[0]
            flag = False
            for id in reset_id:
                seq_id = self.data_id[id]
                curr_steps = self.progress_buf[id]

                # Since stateInit is "Start", we average from 1 to curr_steps (skip index 0 which is never computed)
                if self._max_execution_steps[seq_id] < curr_steps:
                    self._max_execution_steps[seq_id] = curr_steps
                    # Average from index 1 onwards (index 0 is the initial state, no reward computed)
                    self._best_human_pose_error_per_seq[seq_id] = self._human_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    self._best_object_pose_error_per_seq[seq_id] = self._object_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    flag = True
                elif self._max_execution_steps[seq_id] == curr_steps:
                    curr_human_error = self._human_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    curr_object_error = self._object_pose_error_per_seq_step[seq_id, 1:curr_steps].mean()
                    if self._best_human_pose_error_per_seq[seq_id] + self._best_object_pose_error_per_seq[seq_id] > curr_human_error + curr_object_error:
                        self._best_human_pose_error_per_seq[seq_id] = curr_human_error
                        self._best_object_pose_error_per_seq[seq_id] = curr_object_error
                        flag = True

            if (self._max_execution_steps >= 1).all() and flag:
                avg_execution_steps = self._max_execution_steps[self._max_execution_steps > 0].float().mean()
                avg_human_error = self._best_human_pose_error_per_seq[self._best_human_pose_error_per_seq < 1e5].mean()
                avg_object_error = self._best_object_pose_error_per_seq[self._best_object_pose_error_per_seq < 1e5].mean()
                success_count = torch.sum(self._max_execution_steps - (self.max_episode_length - 1) >= 0)
                success_rate = success_count.float() / self.max_episode_length.shape[0]

                print('=' * 60)
                print('EVALUATION METRICS:')
                print(f'  Average Execution Steps: {avg_execution_steps:.2f}')
                print(f'  Average Human Pose Error: {avg_human_error:.4f}')
                print(f'  Average Object Pose Error: {avg_object_error:.4f}')
                print(f'  Success Rate: {success_rate:.2%} ({success_count}/{self.max_episode_length.shape[0]})')
                print('=' * 60)

        if self.reset_buf.sum() > 0 and self.psi > 1:
            reset_ind = (self.reset_buf == 1)
            data_id = self.data_id[reset_ind]
            max_episode_length = self.max_episode_length[data_id]
            if (max_episode_length < self.rollout_length).all():
                self._sum_reward[reset_ind] = 0
                return
            start_index, end_index = self.start_times[reset_ind], self.progress_buf[reset_ind]
            sum_reward = self._sum_reward[reset_ind].mean()
            if torch.rand(1)[0] < 0:
                self._sum_reward[reset_ind] = 0
                return
            self._sum_reward[reset_ind] = 0
            reset_ind = torch.logical_and(reset_ind, self.max_episode_length[self.data_id] > self.rollout_length)
            if reset_ind.sum() < 0.995:
                return
            curr_reward = self._curr_reward[reset_ind]
            state = self._curr_state[reset_ind]
            # Initialize the reward tensor with zeros
            reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            end_i = torch.minimum(max_episode_length, self.rollout_length + start_index)

            assert (end_index < end_i).all()
            # Loop through each example in the batch to assign the values from curr_reward to the correct slices in reward

            # data_num, sample_choice, time, feature

            for i in range(curr_reward.shape[0]):
                if end_index[i] > start_index[i]+30:  # Ensure the indices are valid
                    index_tensor = torch.arange(start_index[i]+10, end_index[i]-10, device=start_index.device)
                    reward[i, data_id[i], start_index[i]+10:end_index[i]-10] = ((end_index[i] - index_tensor) / (end_i[i] - index_tensor))

            adjust_reward, adjust_reward_index = reward.max(dim=0)
            for i in range(reward.shape[1]):
                if self.max_episode_length[i] < self.rollout_length:
                    continue
                for j in range(reward.shape[2]):
                    if self.max_episode_length[i] - j < self.rollout_length:
                        break
                    value, index = self.ref_reward[i, 1:, j].min(dim=0)
                    index = index + 1
                    id1 = adjust_reward_index[i, j]
                    idx = j - start_index[adjust_reward_index[i, j]]

                    if idx > 0 and idx < self.rollout_length and adjust_reward[i, j] > 0.5:
                        self.ref_reward[i, index, j] = adjust_reward[i, j]
                        # state is on GPU; hoi_refs may be on CPU (cpuMotionData) -> match its device
                        self.hoi_refs[i, index, j] = state[id1, idx].to(self.hoi_refs.device)
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 1e-5)
        return

    def compute_hoi_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                          max_episode_length, enable_early_termination, termination_heights, 
                          start_times, rollout_length, reset_ig, contact_reset):

        reset, terminated = self.compute_humanoid_reset(reset_buf, progress_buf, obs_buf, rigid_body_pos,
                                                        max_episode_length, enable_early_termination, termination_heights, 
                                                        start_times, rollout_length)

        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)

        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
        reset = torch.where(reset.bool(), torch.ones_like(reset_buf), terminated)

        if getattr(self, '_term_reason', False):
            self._accumulate_term_reasons(reset, terminated, reset_ig, contact_reset)

        return reset, terminated

    def _accumulate_term_reasons(self, reset, terminated, reset_ig, contact_reset):
        """Tally WHY each episode ended, per body. TERM_REASON=1 only.

        Causes are counted INDEPENDENTLY, not as a partition: a single reset can
        trip several at once (e.g. the humanoid falls and the object diverges in
        the same step), and collapsing that into one 'primary' cause by an
        arbitrary precedence would hide exactly the correlation we're hunting.
        So the row can sum to more than the number of episodes -- that overlap is
        signal, not a bug.
        """
        self._term_steps += 1
        done = reset.bool()
        if not done.any():
            return

        term = terminated.bool()
        fell = self._last_body_fall.bool()      # stashed by compute_humanoid_reset
        nan_ = self._last_invalid_obs.bool()

        cols = [
            done & ~term,                       # completed: ended with no terminal cause = success
            done & term & fell,
            done & term & nan_,
            done & term & reset_ig.bool(),
            done & term & contact_reset.bool(),
        ]
        idx = self._env_subject_idx
        n_bodies = self._term_counts.shape[0]
        # Episodes ended, per body -- the honest denominator. Tracked directly rather
        # than reconstructed from the cause columns, which overlap and can't be summed.
        self._term_episodes += torch.bincount(idx[done], minlength=n_bodies)
        for c, mask in enumerate(cols):
            if mask.any():
                self._term_counts[:, c] += torch.bincount(idx[mask], minlength=n_bodies)

        if self._term_steps % self._term_reason_every == 0:
            self._print_term_reasons()

    def _print_term_reasons(self):
        counts = self._term_counts.cpu().numpy()
        episodes = self._term_episodes.cpu().numpy()
        names = self.subject_bodies if getattr(self, 'subject_bodies', None) else [self.robot_type]
        print('=' * 92)
        print(f'TERMINATION REASONS  (sim step {self._term_steps})')
        print('  % is of episodes ENDED for that body. Causes overlap (one reset can trip')
        print("  several), so a row may exceed 100%. 'completed' = survived the clip = success.")
        hdr = f"{'body':>8s} {'episodes':>9s} " + ' '.join(f'{l:>17s}' for l in self._term_labels)
        print(hdr)
        print('-' * len(hdr))
        for i, nm in enumerate(names):
            n = int(episodes[i])
            if n == 0:
                print(f'{nm:>8s} {0:9d}   (no episodes ended yet)')
                continue
            cells = ' '.join(f'{int(v):7d} ({100.0 * v / n:5.1f}%)' for v in counts[i])
            print(f'{nm:>8s} {n:9d} {cells}')
        print('=' * 92, flush=True)

    # ---- Opt-in reward diagnostics (REWARD_BREAKDOWN=1) -----------------------
    # Periodically prints mean reward TERMS (rb=body, ro=object, rig=interaction,
    # rcg=contact) and the mean reward, grouped by object type, real-vs-synthetic
    # body, beta-cluster, and per-clip contact difficulty. Never affects training
    # (guarded by env var + try/except in the caller). Cadence: REWARD_BREAKDOWN_EVERY
    # steps (default 1000); beta clusters: REWARD_BREAKDOWN_KCLUSTERS (default 4).
    def _kmeans_np(self, X, k, iters=30, seed=0):
        import numpy as np
        rng = np.random.RandomState(seed)
        C = X[rng.choice(len(X), size=k, replace=False)].copy()
        a = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            a = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
            for j in range(k):
                m = a == j
                if m.any():
                    C[j] = X[m].mean(0)
        return a

    def _compute_motion_difficulty(self):
        """Per-motion difficulty (0 easy,1 med,2 hard) from contact_obj (ch 330):
        sustained grip = easy, long object free-flight = hard."""
        fps = float(getattr(self, 'fps_data', 30) or 30)
        buckets = torch.full((len(self.motion_file),), 1, dtype=torch.long, device=self.device)
        for mi, path in enumerate(self.motion_file):
            try:
                x = torch.load(path, map_location='cpu', weights_only=False).detach().float()
                contact = torch.round(x[:, 330])
                cf = float(contact.mean())
                free = (contact < 0.5).tolist()
                run = mx = 0
                for v in free:
                    run = run + 1 if v else 0
                    mx = run if run > mx else mx
                max_ff = mx / fps
                buckets[mi] = 0 if (cf >= 0.85 and max_ff < 0.4) else (2 if (cf < 0.6 or max_ff >= 1.0) else 1)
            except Exception:
                pass
        return buckets

    def _init_reward_breakdown(self):
        import numpy as np, os as _os
        dev, n = self.device, self.num_envs
        no = max(1, len(self.object_name))
        self._rbd_obj_ids = (torch.arange(n, device=dev) % no).long()
        rs = torch.zeros(n, dtype=torch.long, device=dev)
        if hasattr(self, '_body_subject_nums') and hasattr(self, '_env_subject_idx'):
            subj = torch.as_tensor(self._body_subject_nums, device=dev)[self._env_subject_idx]
            rs = (subj >= 100).long()
        self._rbd_rs_ids = rs
        K = int(_os.environ.get('REWARD_BREAKDOWN_KCLUSTERS', '4'))
        if getattr(self, '_env_target_betas', None) is not None and K > 1:
            B = self._env_target_betas.detach().cpu().numpy().astype('float64')
            uniq, inv = np.unique(B.round(4), axis=0, return_inverse=True)
            k = min(K, len(uniq))
            lab = self._kmeans_np(uniq, k)
            self._rbd_cluster_ids = torch.as_tensor(lab[inv], device=dev).long()
            nclust = k
        else:
            self._rbd_cluster_ids = torch.zeros(n, dtype=torch.long, device=dev); nclust = 1
        self._rbd_diff_of_motion = self._compute_motion_difficulty()
        self._rbd_specs = {
            'object':     (self._rbd_obj_ids,     list(self.object_name)),
            'body':       (self._rbd_rs_ids,      ['real', 'synthetic']),
            'beta-clust': (self._rbd_cluster_ids, ['c%d' % i for i in range(nclust)]),
            'difficulty': (None,                  ['easy', 'medium', 'hard']),
        }
        self._rbd_every = int(_os.environ.get('REWARD_BREAKDOWN_EVERY', '1000'))
        self._rbd_reset_accum()
        self._rbd_ready = True
        print("[reward-breakdown] on: %d objects, real/synth=%d/%d, %d beta-clusters, every %d steps"
              % (len(self.object_name), int((rs == 0).sum()), int((rs == 1).sum()), nclust, self._rbd_every), flush=True)

    def _rbd_reset_accum(self):
        self._rbd_sums, self._rbd_cnts = {}, {}
        for g, (ids, names) in self._rbd_specs.items():
            self._rbd_sums[g] = torch.zeros((len(names), 5), device=self.device)  # rb,ro,rig,rcg,reward
            self._rbd_cnts[g] = torch.zeros(len(names), device=self.device)
        self._rbd_steps = 0

    def _log_reward_breakdown(self, rb, ro, rig, rcg):
        if not getattr(self, '_rbd_ready', False):
            self._init_reward_breakdown()
        T = torch.stack([rb, ro, rig, rcg, rb * ro * rig * rcg], dim=1).detach()
        ids_by_g = {'object': self._rbd_obj_ids, 'body': self._rbd_rs_ids,
                    'beta-clust': self._rbd_cluster_ids,
                    'difficulty': self._rbd_diff_of_motion[self.data_id]}
        for g, ids in ids_by_g.items():
            ng = len(self._rbd_specs[g][1])
            self._rbd_cnts[g] += torch.bincount(ids, minlength=ng).float()
            for k in range(5):
                self._rbd_sums[g][:, k] += torch.bincount(ids, weights=T[:, k], minlength=ng)
        self._rbd_steps += 1
        if self._rbd_steps >= self._rbd_every:
            print("\n[reward-breakdown] over %d steps  (rb=body ro=object rig=interaction rcg=contact)" % self._rbd_steps, flush=True)
            for g, (ids, names) in self._rbd_specs.items():
                cnt = self._rbd_cnts[g]; tot = cnt.sum().item()
                print("  by %s:" % g, flush=True)
                for j in torch.argsort(cnt, descending=True).tolist():
                    c = cnt[j].item()
                    if c <= 0:
                        continue
                    m = self._rbd_sums[g][j] / c
                    print("     %-14s %4.0f%%  rb=%.3f ro=%.3f rig=%.3f rcg=%.3f  reward=%.3f"
                          % (names[j], 100 * c / tot, m[0], m[1], m[2], m[3], m[4]), flush=True)
            self._rbd_reset_accum()

    def _compute_reward(self, actions):
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward(self.reward_weights)
        reward = rb
        # Stock object-match terms (object-pose ro, interaction-graph rig, contact-graph
        # rcg). ON by default => stock product rb*ro*rig*rcg. Toggle OFF for objectAug
        # runs where the perturbed object makes them unachievable.
        if self._object_terms_enable:
            reward = reward * ro * rig * rcg
        # Breakdown logging reports the raw TERMS (rb/ro/rig/rcg), so it stays useful
        # even when the object terms are gated out of the product above.
        if os.environ.get('REWARD_BREAKDOWN') == '1':
            try:
                self._log_reward_breakdown(rb, ro, rig, rcg)
            except Exception as _rbde:
                if not getattr(self, '_rbd_warned', False):
                    print("[reward-breakdown] disabled after error: %r" % (_rbde,), flush=True)
                    self._rbd_warned = True
        # Term 1 (opt-in): relative joint-angle pose factor, layered on the product.
        if self._pose_term_enable:
            reward = reward * self._compute_pose_reward()
        # Term 2 (opt-in): relaxed contact / "hold" factor (objectAug companion).
        if self._hold_term_enable:
            reward = reward * self._compute_hold_reward(key_pos, obj_points)
        self.rew_buf[:] = reward
        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        # objectAug / object-terms-off: the perturbed object makes the object/ig/contact
        # resets fire spuriously every step -> relax to human-only termination.
        if self._object_aug or not self._object_terms_enable:
            self.contact_reset = torch.zeros_like(self.contact_reset)
            self.kinematic_reset = human_reset
        index = torch.arange(self._curr_reward.shape[0])
        # # print(self._humanoid_root_states.dtype)
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states,
            self._dof_pos,
            self._dof_vel,
            self._target_states,
        ], dim=1)

        # Track evaluation metrics per sequence (only if evaluation is enabled)
        if self.enable_evaluation:
            # Compute human pose error (mean distance between key body positions)
            human_error = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1)
            # Compute object pose error (mean distance between object point clouds)
            object_error = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1)

            # Store metrics indexed by data_id and progress_buf
            self._human_pose_error_per_seq_step[self.data_id, self.progress_buf] = human_error
            self._object_pose_error_per_seq_step[self.data_id, self.progress_buf] = object_error

        return
    
    def _compute_pose_reward(self):
        """Term 1: parent-relative joint-angle pose matching (opt-in factor).

        Compares simulated vs reference dof_pos -- the 51x3 = 153 parent-relative
        joint DOFs (raw, identical convention on both sides, no heading
        dependence). The SMPL-X DOFs are all bounded hinges (range +/-180deg),
        so they do NOT wrap; the existing energy term already diffs dof values by
        plain subtraction, and we match that (no +/-pi wrap):

            rew_factor = exp(-lambda_pose * sum_j (dof_ref - dof_sim)^2)  in (0, 1]

        multiplied into the reward product when rewardTerms.pose.enable is set.
        """
        dof_sim = self.extract_data_component('dof_pos', obs=self._curr_obs)
        dof_ref = self.extract_data_component('dof_pos', obs=self._curr_ref_obs)
        err = ((dof_ref - dof_sim) ** 2).sum(dim=-1)
        if self._pose_reward_debug:
            # Just-reset envs are state-init'd TO the reference, so their err must
            # be ~0; a large min/median would mean sim/ref dof orderings differ.
            self._posechk_n = getattr(self, '_posechk_n', 0) + 1
            if self._posechk_n % 50 == 1:
                fresh = (self.progress_buf - self.start_times) <= 1
                if bool(fresh.any()):
                    fe = err[fresh]
                    print(f"[posechk] {int(fresh.sum())} fresh: err min={fe.min().item():.4f} "
                          f"med={fe.median().item():.4f} max={fe.max().item():.4f} "
                          f"(small min/med => dof aligned; max = hybrid default-init resets)",
                          flush=True)
        return torch.exp(-self._pose_lambda * err)

    def _compute_hold_reward(self, key_pos, obj_points):
        """Term 2: relaxed contact / 'hold' factor (opt-in objectAug companion).

        Ported from objectaug-experiment, then gated on the REFERENCE grip. The
        'keep a grip' pressure is applied to a hand ONLY on frames where the source
        clip has that hand in contact with the object. When the reference isn't
        holding with a hand -- object in free-flight, at rest on a surface, or the
        hand simply idle -- that hand's factor is a neutral 1. Otherwise the
        proximity term would wrongly drag the wrists onto the object during the very
        frames the source released it (the release/no-contact frames matter for
        manipulation quality). This is NOT a replacement for rcg/ro/rig; it layers.

        Per hand, when the reference holds with it, two soft sub-factors in (0,1]:
          proximity : wrist near the (scaled, posed) object surface, exp(-lambda*d).
          contact   : that hand is in contact now, floored to [0.5,1] so it shapes
                      rather than starves.
        Weight = rewardTerms.hold.lambda.
        """
        if not hasattr(self, '_hand_key_ids'):
            # order preserved -> [L_Wrist, R_Wrist], aligned with the (left, right)
            # hand-link id groups in the loop below.
            self._hand_key_ids = [i for i, n in enumerate(self.key_bodies)
                                  if n in ('L_Wrist', 'R_Wrist')]
        hand_pos = key_pos[:, self._hand_key_ids, :]                 # (E, 2, 3)
        min_d = torch.cdist(hand_pos, obj_points).min(dim=-1)[0]     # (E, 2) per wrist
        r_prox = torch.exp(-self._hold_lambda * min_d)               # (E, 2) per wrist

        contact_thres = 0.1
        ref_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        live_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        factors = []
        for h, ids in enumerate((list(range(17, 33)), list(range(36, 52)))):  # left, right hand links
            ref_any = (ref_contact[:, ids] > contact_thres).any(dim=-1).float()   # ref holds w/ this hand
            live_any = (live_contact[:, ids] > contact_thres).any(dim=-1).float()
            shaped = r_prox[:, h] * (0.5 + 0.5 * live_any)          # grip shaping in (0,1]
            # only shape when the reference holds with this hand; else neutral 1.
            factors.append(ref_any * shaped + (1.0 - ref_any))
        return torch.stack(factors, dim=-1).mean(dim=-1)           # (E,) in (0,1]

    def compute_humanoid_reward(self, w):
        # body pos reward
        len_keypos = len(self._key_body_ids)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1

        pos_diff = ref_key_pos - key_pos
        if self._body_normalized_reward and self._env_body_height is not None:
            # Divide per-env by body height so a 10cm error on a 1.4m body
            # contributes the same as 10cm on a 1.7m body. Broadcast height
            # (num_envs,) -> (num_envs, num_key_bodies, 3) via view+unsqueeze.
            pos_diff = pos_diff / self._env_body_height.view(-1, 1, 1)
        ep = torch.mean((pos_diff**2).sum(dim=-1) * weight_hp[:, self._key_body_ids],dim=-1)
        rp = torch.exp(-ep*w['p'])

        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er*w['r'])
        
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        # body pos vel reward
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
        rpv = torch.exp(-epv*w['pv'])

        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        # body rot vel reward
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # energy penalty
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

        rb = rp*rr*rpv*rrv*energy
        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        
        return rb, human_reset, key_pos, ref_key_pos
    
    def compute_obj_reward(self, w):
        # object pos reward
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        
        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

        local_obj_rot = quat_mul(heading_rot, obj_rot)

        object_points = self.object_points[self.object_id[self.data_id]] * self._obj_pts_scale
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)

        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)

        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1) # * (1 - weight_h.max(dim=-1)[0])
        rop = torch.exp(-eop*w['op'])

        # object rot reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        
        eor = torch.mean(diff,dim=-1)
        ror = torch.exp(-eor*w['or'])

        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        # object pos vel reward
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
        ropv = torch.exp(-eopv*w['opv'])

        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        # object rot vel reward
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
        rorv = torch.exp(-eorv*w['orv'])
        
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)
        
        obj_energy = (obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
        ro = rop*ror*ropv*rorv*obj_energy
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        return ro, object_reset, obj_points, ref_obj_points
    
    def compute_ig_reward(self, w, key_pos, ref_key_pos, obj_points, ref_obj_points):
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1,len_keypos,3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1,len_keypos,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        ### interaction graph reward ###
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

        rig = torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))

        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        return rig, reset_ig
    
    def compute_cg_reward(self, w):    
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        left_contact_hand_ids = list(range(17, 33))
        
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)


        right_contact_hand_ids = list(range(36, 52))
        
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)
        
        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right

        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*w['cg_other'])
        
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*w['cg_all'])

        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        return rcg, contact_reset
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            # render_all_clips forces every env onto ONE specific clip so we can
            # export a deterministic per-clip video; the normal replay picks a
            # random clip per env (matched to that env's object).
            if getattr(self, '_force_clip_id', None) is not None:
                self.data_id = torch.full((self.num_envs,), int(self._force_clip_id),
                                          device=self.device, dtype=torch.long)
            else:
                self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        t = to_torch(
                [
                    t if t < self.max_episode_length[self.data_id[i]] else self.max_episode_length[self.data_id[i]]-1
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long
            )
        ### update object ###
        self._target_states[env_ids, :3] = self.extract_data_component('obj_pos', True, self.data_id[env_ids], t)
        self._target_states[env_ids, 3:7] = self.extract_data_component('obj_rot', True, self.data_id[env_ids], t)
        self._target_states[env_ids, 7:10] = torch.zeros_like(self._target_states[env_ids, 7:10])
        self._target_states[env_ids, 10:13] = torch.zeros_like(self._target_states[env_ids, 10:13])

        ### update subject ###   
        _humanoid_root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
        _humanoid_root_rot = self.extract_data_component('root_rot', True, self.data_id[env_ids], t)
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])
        
        self._dof_pos[env_ids] = self.extract_data_component('dof_pos', True, self.data_id[env_ids], t)
        self._dof_vel[env_ids] = self.extract_data_component('dof_vel', True, self.data_id[env_ids], t)


        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._refresh_sim_tensors()
        obj_contact = self.extract_data_component('contact_obj', True, self.data_id[env_ids], t)
        obj_contact = torch.any(obj_contact > 0.1, dim=-1)
        human_contact = self.extract_data_component('contact_human', True, self.data_id[env_ids], t)
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]

                if obj_contact[env_id] == True:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                else:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(0., 0., 1.))
                    
                handle = self.humanoid_handles[env_id]
                for j in range(self.num_bodies):
                    if human_contact[env_id, j] > 0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)
        self.gym.simulate(self.sim)
        # --- capture frame ---
        # Skipped while render_all_clips is driving (it captures per-clip to mp4
        # itself); otherwise this dumps the env-0 replay to replay_frames/*.png.
        if hasattr(self, '_video_writer') and not getattr(self, '_render_clips_active', False):
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.envs[0], self._video_cam, gymapi.IMAGE_COLOR)
            img = img.reshape(self._video_height, self._video_width, 4)[..., :3]
            imageio.imwrite(f'replay_frames/frame_{self._video_frame_idx:06d}.png', img); self._video_frame_idx += 1
        # end of mod part

        return

    def render_all_clips(self, out_dir, lo=0, hi=None, fps=30):
        """Export ONE replay mp4 per motion clip in [lo, hi) in a single launch.

        For each clip it forces env 0 onto that clip (via _force_clip_id), steps
        through the clip's full length with play_dataset_step, and encodes env 0's
        camera to out_dir/<clipname>.mp4. Because each env's OBJECT MESH is fixed
        at creation (env e -> object e % num_objects), env 0 only renders the
        right object when the run is filtered to a single object -- so the driver
        launches this once per object (num_envs=1, dataObjects=[obj]).
        """
        import imageio
        if not hasattr(self, '_video_cam'):
            raise RuntimeError(
                "render_all_clips needs the play_dataset camera -- launch with "
                "--play_dataset and WITHOUT RECORD_VIDEO set.")
        hi = self.num_motions if hi is None else min(int(hi), self.num_motions)
        os.makedirs(out_dir, exist_ok=True)
        self._render_clips_active = True   # suppress the per-step replay_frames dump
        try:
            for m in range(int(lo), hi):
                base = os.path.basename(self.motion_file[m])
                name = base[:-3] if base.endswith('.pt') else base
                path = os.path.join(out_dir, name + '.mp4')
                length = int(self.max_episode_length[m])
                # Skip clips already rendered so a timeout / re-run RESUMES instead
                # of restarting from scratch. (A partial/corrupt tail file is rare
                # since writer.close() is the last step; delete it to re-render.)
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    print(f"[render] {m - int(lo) + 1}/{hi - int(lo)}  {name}  "
                          f"exists, skipping", flush=True)
                    continue
                self._force_clip_id = m
                writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                            quality=8, macro_block_size=None)
                for t in range(length):
                    self.play_dataset_step(t)
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    img = self.gym.get_camera_image(self.sim, self.envs[0],
                                                    self._video_cam, gymapi.IMAGE_COLOR)
                    img = img.reshape(self._video_height, self._video_width, 4)[..., :3]
                    writer.append_data(img)
                writer.close()
                print(f"[render] {m - int(lo) + 1}/{hi - int(lo)}  {name}  "
                      f"{length} frames -> {path}", flush=True)
        finally:
            self._force_clip_id = None
            self._render_clips_active = False
    

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[-1][6:-3]
                images_dir = resolve_data_path("images", dataname, must_exist=False)
                images_dir.mkdir(parents=True, exist_ok=True)
                rgb_filename = images_dir / ("rgb_env%d_frame%05d.png" % (env_ids, frame_id))
                self.gym.write_viewer_image_to_file(self.viewer, str(rgb_filename))
        return

    def print_final_eval_summary(self):
        """Print final evaluation summary at the end of inference"""
        if not self.enable_evaluation:
            return

        evaluated_mask = self._max_execution_steps >= 1
        num_evaluated = evaluated_mask.sum()

        if num_evaluated == 0:
            print("=" * 60)
            print("WARNING: No sequences were evaluated!")
            print("Consider increasing --max_steps in the evaluation script")
            print("=" * 60)
            return

        avg_execution_steps = self._max_execution_steps[evaluated_mask].float().mean()
        avg_human_error = self._best_human_pose_error_per_seq[evaluated_mask].mean()
        avg_object_error = self._best_object_pose_error_per_seq[evaluated_mask].mean()
        success_count = torch.sum(self._max_execution_steps[evaluated_mask] - (self.max_episode_length[evaluated_mask] - 1) >= 0)
        success_rate = success_count.float() / num_evaluated

        # Visit statistics
        min_visits = self._sequence_visit_count.min().item()
        max_visits = self._sequence_visit_count.max().item()
        avg_visits = self._sequence_visit_count.float().mean().item()

        print("\n" + "=" * 60)
        print("FINAL EVALUATION SUMMARY:")
        print(f"  Sequences Evaluated: {num_evaluated}/{self.num_motions} ({100.0 * num_evaluated / self.num_motions:.1f}%)")
        print(f"  Sequence Visits - Min: {min_visits}, Max: {max_visits}, Avg: {avg_visits:.1f}")
        print(f"  Average Execution Steps: {avg_execution_steps:.2f}")
        print(f"  Average Human Pose Error: {avg_human_error:.4f}")
        print(f"  Average Object Pose Error: {avg_object_error:.4f}")
        print(f"  Success Rate: {success_rate:.2%} ({success_count}/{num_evaluated})")
        print("=" * 60 + "\n")
    
@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat
