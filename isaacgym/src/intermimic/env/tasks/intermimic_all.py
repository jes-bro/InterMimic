import torch
from .intermimic import InterMimic

from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from ...learning import (intermimic_network_builder,
                         intermimic_transformer_network_builder,
                         intermimic_models_teacher)
from ...utils.path_utils import resolve_data_path, resolve_repo_path
from ...utils.per_body_refs import per_body_reference_files
from torch.func import vmap
from functorch import make_functional
import os
import yaml

def get_all_paths(dir_path):
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths

class InterMimic_All(InterMimic):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        # Read transformer observation flag (only used for student policy)
        self.use_transformer_obs = cfg['env'].get('useTransformerObs', False)
        self.action_buf = torch.zeros(
            (self.num_envs, 153), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros(
            (self.num_envs, 153), device=self.device, dtype=torch.float)
        self.obs_buf_retarget = torch.zeros(
            (self.num_envs, cfg["env"]["numObsRetarget"]), device=self.device, dtype=torch.float)
        self.models = []
        self.running_means = []
        self.running_vars = []
        obs_shape = cfg["env"]["numObs"]
        config = {
            'actions_num' : 153,
            'input_shape' : (obs_shape, ),
            'num_seqs' : cfg["env"]["numEnvs"] * 1,
            'value_size': 1,
        }
        teacher_cfg_path = resolve_data_path("cfg", "train", "rlg", os.path.basename(cfg["env"]["teacherPolicyCFG"]))
        with open(teacher_cfg_path, 'r') as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)

        # Pick the teacher network builder from the teacher's OWN train cfg. The
        # source-teachers are transformers (intermimic_transformer); the paper's
        # per-subject teachers were MLP (intermimic). No silent default -- an
        # unknown arch loads wrong weights and produces silently-bad targets.
        _teacher_net = cfg_teacher['params']['network']['name']
        if _teacher_net == 'intermimic_transformer':
            network = intermimic_transformer_network_builder.InterMimicBuilder()
        elif _teacher_net == 'intermimic':
            network = intermimic_network_builder.InterMimicBuilder()
        else:
            raise ValueError(
                f"InterMimic_All: unsupported teacher network '{_teacher_net}' in "
                f"{teacher_cfg_path} (expected 'intermimic' or 'intermimic_transformer')")
        network.load(cfg_teacher['params']['network'])
        network = intermimic_models_teacher.ModelInterMimicContinuous(network)
        teacher_policy = resolve_repo_path(cfg["env"]["teacherPolicy"])
        models_path = get_all_paths(str(teacher_policy))
        self.models = []
        self.functional_models = []
        self.params_list = []
        self.running_means = []
        self.running_vars = []
        self.models_subid = []
        for model_path in models_path:
            subid = int(model_path.split('/')[-1].split('.')[0][3:])
            ck = torch_ext.load_checkpoint(model_path)
            model = network.build(config)
            model.to(self.device)
            model.load_state_dict(ck['model'])
            self.models.append(model)
            f_model, params = make_functional(model)
            self.functional_models.append(f_model)
            self.params_list.append(params)
            running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
            self.running_means.append(running_mean)
            self.running_vars.append(running_var)
            self.models_subid.append(subid)
    
        # Transpose list of parameter tuples into tuple of parameter lists
        self.params_zip = list(zip(*self.params_list))
        # Now stack along a new dimension (model dimension = 0)
        self.stacked_params = tuple(torch.stack(p_tensors, dim=0) for p_tensors in self.params_zip)
        self.running_means_all = torch.stack(self.running_means).float()  # shape [num_models, ...]
        self.running_vars_all = torch.stack(self.running_vars).float()   # shape [num_models, ...]
        
        self.motion_file_retarget = cfg['env']['motion_file_retarget']
        motion_file_retarget = os.listdir(self.motion_file_retarget)
        motion_file_retarget = sorted([data_path for data_path in motion_file_retarget if data_path.split('_')[0] in cfg['env']['dataSub']])
        self.motion_file_retarget = [os.path.join(self.motion_file_retarget, data_path) for data_path in motion_file_retarget]
        self.num_motions = len(self.motion_file_retarget)
        self.dataset_index = to_torch([int(data_path.split('/')[-1].split('_')[0][3:]) for data_path in self.motion_file_retarget], dtype=torch.long, device=self.device)
        # self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file if data_path.split('_')[0] in ['sub9']])
        # super().__init__(cfg=cfg,
        #                         sim_params=sim_params,
        #                         physics_engine=physics_engine,
        #                         device_type=device_type,
        #                         device_id=device_id,
        #                         headless=headless)
        # --- OPTION A: score this policy against OUR per-body reference -----------
        # retargetedMotionDir (the g3 convention, <tree>/<body>/<clip>.pt) makes this
        # task read the SAME reference the g3 students are scored against, for the one
        # body an eval pair runs. WITHOUT the key nothing changes -- the paper's own
        # pairing (motion_file raw + motion_file_retarget = the authors' tree) is
        # untouched, so every existing InterMimic_All run is byte-identical.
        #
        # WHY BOTH LISTS. Here motion_file feeds hoi_data (what the metrics score
        # against and what resets seed from) and motion_file_retarget feeds the
        # policy's observation. The g3 task has ONE reference for both (intermimic.py
        # replaces motion_file with the per-body files), so matching it means pointing
        # both at our per-body clip.
        #
        # WHY THE CLIP LIST COMES FROM motion_file. The authors' retarget tree covers
        # 2,209 clips; OMOMO_new (what the g3 students are scored on) has 3,356.
        # Listing from motion_file puts this policy on the SAME clips as the students.
        #
        # ONE BODY ONLY: an eval runs one (body, source) pair (eval_per_pair.py passes
        # --subject_bodies), so there is no body-major expansion and object_id /
        # dataset_index stay clip-length. More than one body is refused rather than
        # silently mis-indexed; training with this key is not supported.
        retarget_dir = cfg['env'].get('retargetedMotionDir', None)
        if retarget_dir:
            bodies = list(cfg['env'].get('subjectBodies') or [])
            src_dir = cfg['env']['motion_file']
            files, clip_names = per_body_reference_files(retarget_dir, src_dir,
                                                         cfg['env']['dataSub'], bodies)
            motion_file_retarget = clip_names            # basenames, for object_name below
            self.motion_file_retarget = files
            self.num_motions = len(files)
            self.dataset_index = to_torch([int(n.split('_')[0][3:]) for n in clip_names],
                                          dtype=torch.long, device=self.device)
            print(f"[retarget] InterMimic_All scored against OUR reference: {len(files)} clips of "
                  f"body {bodies[0]} from '{retarget_dir}' (observation AND metrics; clip list "
                  f"from {src_dir})", flush=True)

        object_name = [motion_example.split('_')[-2] for motion_example in self.motion_file_retarget]
        if retarget_dir:
            self.motion_file = list(self.motion_file_retarget)   # one reference for both (see above)
        else:
            self.motion_file = [os.path.join(cfg['env']['motion_file'], data_path) for data_path in motion_file_retarget]
        self.object_id = to_torch([self.object_name.index(name) for name in object_name], dtype=torch.long, device=self.device)
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(self.object_name))], dim=0)
        self.hoi_data = self._load_motion(self.motion_file, startk=1)
        self.hoi_data_retarget = self._load_motion(self.motion_file_retarget, startk=1)
        return

    
    def _compute_observations_retarget(self, env_ids=None):
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data_retarget[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            if self.use_transformer_obs:
                # Transformer: 4 time steps (0, 1, 4, 16) for student policy
                obs_t0 = self._compute_observations_iter(self.hoi_data_retarget, None, 0)
                obs_t1 = self._compute_observations_iter(self.hoi_data_retarget, None, 1)
                obs_t4 = self._compute_observations_iter(self.hoi_data_retarget, None, 4)
                obs_t16 = self._compute_observations_iter(self.hoi_data_retarget, None, 16)
                self.obs_buf_retarget[:] = torch.cat((obs_t0, obs_t1, obs_t4, obs_t16), dim=-1)
            else:
                # MLP: 2 time steps (1, 16)
                self.obs_buf_retarget[:] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, None, 1), self._compute_observations_iter(self.hoi_data_retarget, None, 16)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data_retarget[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            if self.use_transformer_obs:
                # Transformer: 4 time steps (0, 1, 4, 16) for student policy
                obs_t0 = self._compute_observations_iter(self.hoi_data_retarget, env_ids, 0)
                obs_t1 = self._compute_observations_iter(self.hoi_data_retarget, env_ids, 1)
                obs_t4 = self._compute_observations_iter(self.hoi_data_retarget, env_ids, 4)
                obs_t16 = self._compute_observations_iter(self.hoi_data_retarget, env_ids, 16)
                self.obs_buf_retarget[env_ids] = torch.cat((obs_t0, obs_t1, obs_t4, obs_t16), dim=-1)
            else:
                # MLP: 2 time steps (1, 16)
                self.obs_buf_retarget[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, env_ids, 1), self._compute_observations_iter(self.hoi_data_retarget, env_ids, 16)), dim=-1)

        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return


    def single_model_forward(self, params, obs, mean, var):
        curr_obs = (obs - mean) / torch.sqrt(var + 1e-5)
        curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
        
        # Construct input_dict for the model
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs': curr_obs,
            'rnn_states': None
        }
        
        # Run the functional model
        res_dict = self.functional_models[0](params, input_dict)  # We'll rely on vmap to pick correct slice of params
        mu = res_dict['mus']
        sigma = res_dict['sigmas']
        return mu, sigma


    def step(self, weights):

        self.pre_physics_step(weights)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        with torch.no_grad():
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            mus_all, sigma_all = batched_forward(self.stacked_params, self.obs_buf.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1), self.running_means_all, self.running_vars_all)
            distr = torch.distributions.Normal(mus_all, sigma_all)
            selected_action = distr.sample()
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)


    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        id_to_index = {subid: i for i, subid in enumerate(self.models_subid)}
        self.model_indices = torch.tensor([id_to_index[subid.item()] for subid in self.dataset_id], 
                             dtype=torch.long, device=self.device)
        N = self.dataset_id.shape[0]
        self.sample_indices = torch.arange(N, device=self.device)
        with torch.no_grad():
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            mus_all, sigma_all = batched_forward(self.stacked_params, self.obs_buf.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1), self.running_means_all, self.running_vars_all)
            distr = torch.distributions.Normal(mus_all, sigma_all)
            selected_action = distr.sample()
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]
        return
    
    def post_physics_step(self):
        self.progress_buf += 1
                
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._update_hist_hoi_obs()
        self._compute_hoi_observations()
        self._compute_observations(env_ids)
        self._compute_observations_retarget(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return
