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

import os
import torch, time

try:
    import imageio
    from isaacgym import gymapi
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from . import common_player

class InterMimicPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', False)
        
        super().__init__(config)
        return

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life * 10
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        _record_path = os.environ.get("RECORD_VIDEO")
        _max_video_frames = int(os.environ.get("MAX_VIDEO_FRAMES", "1000"))
        _frames_written = 0
        _writer = None
        _cam_handle = None
        _cam_props = None
        if _record_path and _IMAGEIO_AVAILABLE:
            task = self.env.task

            # Camera placement is configurable so you can switch between
            # "single env close-up" and "wide grid view".
            # RECORD_VIDEO_CAM_POS / RECORD_VIDEO_CAM_TARGET: comma-separated "x,y,z"
            # RECORD_VIDEO_WIDE=1: use a preset wide overhead view of all envs
            def _parse_vec3(s, default):
                if not s:
                    return default
                try:
                    x, y, z = [float(v) for v in s.split(",")]
                    return (x, y, z)
                except Exception:
                    return default

            if os.environ.get("RECORD_VIDEO_WIDE", "0") == "1":
                _default_pos = (15.0, 15.0, 12.0)
                _default_target = (0.0, 0.0, 1.0)
            else:
                _default_pos = (3.0, 3.0, 2.5)
                _default_target = (0.0, 0.0, 1.0)
            _cam_pos = _parse_vec3(os.environ.get("RECORD_VIDEO_CAM_POS"), _default_pos)
            _cam_target = _parse_vec3(os.environ.get("RECORD_VIDEO_CAM_TARGET"), _default_target)
            _record_env_idx = int(os.environ.get("RECORD_VIDEO_ENV_IDX", "0"))

            _cam_props = gymapi.CameraProperties()
            _cam_props.width = 1280
            _cam_props.height = 720
            _cam_handle = task.gym.create_camera_sensor(task.envs[_record_env_idx], _cam_props)
            task.gym.set_camera_location(
                _cam_handle, task.envs[_record_env_idx],
                gymapi.Vec3(*_cam_pos),
                gymapi.Vec3(*_cam_target),
            )
            _writer = imageio.get_writer(_record_path, fps=30, codec="libx264", quality=8)
            print(f"[player] recording video to {_record_path} (cap {_max_video_frames} frames, "
                  f"env[{_record_env_idx}] cam pos {_cam_pos} -> {_cam_target})")

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            if self.env.task.play_dataset:
                # play dataset
                while True:
                    for t in range(self.env.task.max_episode_length.max()): 
                        self.env.task.play_dataset_step(t) 
            else:
                # inference
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)
                    obs_dict, r, done, info =  self.env_step(self.env, action)
                    cr += r
                    steps += 1

                    self._post_step(info)

                    if _writer is not None:
                        task = self.env.task
                        task.gym.step_graphics(task.sim)
                        task.gym.render_all_camera_sensors(task.sim)
                        img = task.gym.get_camera_image(
                            task.sim, task.envs[0], _cam_handle, gymapi.IMAGE_COLOR
                        )
                        img = img.reshape(_cam_props.height, _cam_props.width, 4)[..., :3]
                        _writer.append_data(img)
                        _frames_written += 1
                        if _frames_written >= _max_video_frames:
                            _writer.close()
                            print(f"[player] wrote {_frames_written} frames to {_record_path}, video done")
                            _writer = None

                    if render:
                        self.env.render(mode = 'human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)
                        sum_game_res += game_res
                        if batch_size//self.num_agents == 1 or games_played >= n_games:
                            break
                    
                    done_indices = done_indices[:, 0]

        # Print final evaluation summary if evaluation is enabled
        if hasattr(self.env.task, 'print_final_eval_summary'):
            self.env.task.print_final_eval_summary()

        if _writer is not None:
            _writer.close()
            print(f"[player] wrote video to {_record_path}")

        return

    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        return
