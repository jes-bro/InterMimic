"""
Headless joint_monkey: records DOF animation to mp4 instead of opening a viewer.
Run on a cluster node with a GPU; no X server needed.
"""

import math
import numpy as np
import imageio
from isaacgym import gymapi, gymutil


def clamp(x, lo, hi):
    return max(min(x, hi), lo)


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
]

args = gymutil.parse_arguments(
    description="Headless joint monkey: record DOF animation",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0},
        {"name": "--speed_scale", "type": float, "default": 1.0},
        {"name": "--output", "type": str, "default": "joint_monkey.mp4"},
        {"name": "--max_frames", "type": int, "default": 1500},
        {"name": "--width", "type": int, "default": 1280},
        {"name": "--height", "type": int, "default": 720},
    ])

# --- sim setup ---
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
sim_params.use_gpu_pipeline = False

# IMPORTANT: graphics_device_id must NOT be -1, or camera rendering is disabled.
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)
assert sim is not None, "Failed to create sim"

gym.add_ground(sim, gymapi.PlaneParams())

# --- asset ---
asset_root = "../../assets"
asset_file = asset_descriptors[args.asset_id].file_name
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]
dof_positions = dof_states['pos']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i], upper_limits[i] = -math.pi, math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            lower_limits[i], upper_limits[i] = -1.0, 1.0
    dof_positions[i] = defaults[i]
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]),
                                             0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# --- envs (smaller grid so the camera frames it nicely) ---
num_envs = 9
num_per_row = 3
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs, actor_handles = [], []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    actor = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

# --- camera sensor (replaces the viewer) ---
cam_props = gymapi.CameraProperties()
cam_props.width = args.width
cam_props.height = args.height
cam_handle = gym.create_camera_sensor(envs[0], cam_props)
# attach to env[0] so transform is in that env's local frame; tweak as you like
gym.set_camera_location(
    cam_handle, envs[0],
    gymapi.Vec3(6.0, 4.0, 6.0),
    gymapi.Vec3(0.0, 1.0, 0.0),
)

# --- animation state ---
ANIM_SEEK_LOWER, ANIM_SEEK_UPPER, ANIM_SEEK_DEFAULT, ANIM_FINISHED = 1, 2, 3, 4
anim_state = ANIM_SEEK_LOWER
current_dof = 0
print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)

frames_done = 0
finished_all_dofs = False
while frames_done < args.max_frames and not finished_all_dofs:
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]
    if anim_state == ANIM_SEEK_LOWER:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    if anim_state == ANIM_SEEK_DEFAULT:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = defaults[current_dof]
        next_dof = (current_dof + 1) % num_dofs
        if next_dof == 0:
            finished_all_dofs = True
        current_dof = next_dof
        anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

    # render to the camera sensor instead of a viewer window
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = gym.get_camera_image(sim, envs[0], cam_handle, gymapi.IMAGE_COLOR)
    # IMAGE_COLOR returns HxW*4 uint8 (RGBA flattened); reshape and drop alpha
    img = img.reshape(args.height, args.width, 4)[..., :3]
    writer.append_data(img)
    frames_done += 1

writer.close()
print(f"Wrote {frames_done} frames to {args.output}")
gym.destroy_sim(sim)
