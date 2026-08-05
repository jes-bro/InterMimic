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
        # Leading frames to discard before writing any. Counted separately from
        # _frames_written so MAX_VIDEO_FRAMES still means the length of the
        # finished video rather than the length before trimming.
        _skip_video_frames = int(os.environ.get("SKIP_VIDEO_FRAMES", "0"))
        _skipped = 0
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

        # --- DUMP_TRAJ: record the humanoid's per-frame GLOBAL body state so the
        # SMPL-X SURFACE can be posed offline to reproduce exactly what the policy
        # did in sim (scripts/smplx_pose.py pose_from_bodies + a renderer). Saves
        # body_rot/body_pos (52 bodies) + object pose to an npz, then exits. Env0.
        _dump_path = os.environ.get("DUMP_TRAJ")
        _dump_max = int(os.environ.get("DUMP_FRAMES", os.environ.get("MAX_VIDEO_FRAMES", "1000")))
        _traj = {"body_rot": [], "body_pos": [], "obj_pos": [], "obj_rot": []} if _dump_path else None
        if _traj is not None:
            _t = self.env.task
            _sb = getattr(_t, "subject_bodies", None)
            _traj_subject = (_sb[int(_t._env_subject_idx[0])] if _sb else str(getattr(_t, "robot_type", "unknown")))
            print(f"[player] DUMP_TRAJ -> {_dump_path} (subject={_traj_subject}, cap {_dump_max} frames)", flush=True)

        def _dump_step():
            t = self.env.task
            _traj["body_rot"].append(t._rigid_body_rot[0].detach().cpu().numpy().copy())
            _traj["body_pos"].append(t._rigid_body_pos[0].detach().cpu().numpy().copy())
            _traj["obj_pos"].append(t._target_states[0, 0:3].detach().cpu().numpy().copy())
            _traj["obj_rot"].append(t._target_states[0, 3:7].detach().cpu().numpy().copy())

        def _dump_save():
            import numpy as _np
            _np.savez_compressed(
                _dump_path,
                body_rot=_np.asarray(_traj["body_rot"], dtype=_np.float32),
                body_pos=_np.asarray(_traj["body_pos"], dtype=_np.float32),
                obj_pos=_np.asarray(_traj["obj_pos"], dtype=_np.float32),
                obj_rot=_np.asarray(_traj["obj_rot"], dtype=_np.float32),
                subject=_traj_subject)
            print(f"[player] DUMP_TRAJ wrote {len(_traj['body_rot'])} frames -> {_dump_path}", flush=True)

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
                # Per-clip video export: if RENDER_CLIPS_OUT is set, render one
                # mp4 per clip and stop (a whole-object batch in one launch).
                _render_out = os.environ.get("RENDER_CLIPS_OUT")
                if _render_out:
                    self.env.task.render_all_clips(
                        _render_out,
                        lo=int(os.environ.get("RENDER_CLIPS_LO", 0)),
                        hi=(int(os.environ["RENDER_CLIPS_HI"])
                            if os.environ.get("RENDER_CLIPS_HI") else None),
                        fps=int(os.environ.get("RENDER_CLIPS_FPS", 30)))
                    return
                # play dataset
                # No-silent-fallback: RECORD_VIDEO was requested but no writer
                # exists -> imageio (or imageio-ffmpeg) is missing in this env.
                # Replaying with no video and "succeeding" hides the failure, so
                # fail loudly instead of producing a video-less run.
                if _record_path is not None and _writer is None:
                    raise RuntimeError(
                        f"RECORD_VIDEO={_record_path} is set but no video writer "
                        "was created (imageio unavailable in this conda env). "
                        "Install it, e.g.:  pip install imageio imageio-ffmpeg  "
                        "into the env the replay script activates (intermimic-gym2).")

                # One full pass over the loaded clip(s). When recording we keep
                # replaying until _max_video_frames frames are captured, but we
                # ALWAYS guarantee forward progress: if a pass captures zero new
                # frames (e.g. max_episode_length == 0 for a clip that loaded with
                # zero length), stop instead of spinning the while loop forever
                # with no output. This is the bug that made replays hang and emit
                # no mp4.
                _pass_len = int(self.env.task.max_episode_length.max())
                while True:
                    _before = _frames_written
                    for t in range(_pass_len):
                        self.env.task.play_dataset_step(t)
                        # Record the sim's own body state. Same loop as the video
                        # so a dump and a recording always describe the same
                        # frames, and either can be produced without the other.
                        if _traj is not None and len(_traj["body_rot"]) < _dump_max:
                            _dump_step()
                        # Capture frame to RECORD_VIDEO if set (mirror of the
                        # inference branch's recording loop below).
                        if _writer is not None:
                            task = self.env.task
                            task.gym.step_graphics(task.sim)
                            task.gym.render_all_camera_sensors(task.sim)
                            img = task.gym.get_camera_image(
                                task.sim, task.envs[0], _cam_handle, gymapi.IMAGE_COLOR
                            )
                            img = img.reshape(_cam_props.height, _cam_props.width, 4)[..., :3]
                            # play_dataset_step writes the root and dof tensors,
                            # but they only reach the sim on the following step,
                            # so the first image still shows the asset's default
                            # T-pose rather than frame 0 of the motion.
                            # SKIP_VIDEO_FRAMES drops those leading frames.
                            if _skipped < _skip_video_frames:
                                _skipped += 1
                            else:
                                _writer.append_data(img)
                                _frames_written += 1
                                if _frames_written >= _max_video_frames:
                                    break
                    if _writer is None:
                        break  # not recording -> one pass is enough, don't loop
                    # Recording: done when full, OR when a pass added nothing (no
                    # frames left to capture) so we never loop forever.
                    if _frames_written >= _max_video_frames or _frames_written == _before:
                        break
                    if _traj is not None and len(_traj["body_rot"]) >= _dump_max:
                        break

                if _writer is not None:
                    _writer.close()
                    _writer = None
                    if _frames_written == 0:
                        # Loud, not a silent empty/zero-byte mp4.
                        raise RuntimeError(
                            f"RECORD_VIDEO={_record_path}: captured 0 frames "
                            f"(max_episode_length.max()={_pass_len}). The clip(s) "
                            "loaded with zero length -- check dataSub / motion_file "
                            "/ CLIP so a real motion is selected.")
                    print(f"[player] wrote {_frames_written} frames to {_record_path}, video done", flush=True)

                if _traj is not None:
                    _dump_save()

                # play_dataset is a finite replay: when recording OR dumping,
                # exit the whole n_games loop -- don't fall through and keep
                # replaying. Dumping was previously not a reason to stop, so a
                # DUMP_TRAJ run with no RECORD_VIDEO replayed games_num *
                # n_game_life * 10 times and never reached the save.
                if _record_path is not None or _dump_path is not None:
                    import sys
                    sys.exit(0)
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
                    # A policy rollout dumps the same way a replay does, so the
                    # two render identically and can be compared frame to frame.
                    if _traj is not None:
                        if len(_traj["body_rot"]) < _dump_max:
                            _dump_step()
                        else:
                            _dump_save()
                            import sys
                            sys.exit(0)
                    cr += r
                    steps += 1

                    self._post_step(info)

                    if _traj is not None:
                        _dump_step()
                        if len(_traj["body_rot"]) >= _dump_max:
                            _dump_save()
                            import sys
                            sys.exit(0)

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
                            # When recording was the whole point (RECORD_VIDEO set),
                            # exit now — don't keep running the policy and printing
                            # reward stats that block the next render in the script.
                            import sys
                            sys.exit(0)

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
        else:
            # fn == 'Base' means NO checkpoint was loaded: the policy net keeps
            # its random init. Any rollout/eval metrics are then meaningless but
            # look real. Warn loudly (stderr) so this can't pass unnoticed -- the
            # only legitimate no-checkpoint case is --play_dataset mocap replay,
            # where the policy isn't used.
            import sys as _sys
            print("\n" + "!" * 70 +
                  "\n[player] WARNING: no checkpoint restored (checkpoint='Base').\n"
                  "[player] The policy is running on RANDOM initial weights.\n"
                  "[player] Any success-rate / pose-error metrics are MEANINGLESS\n"
                  "[player] unless this is a --play_dataset mocap replay.\n" +
                  "!" * 70 + "\n", file=_sys.stderr, flush=True)
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
