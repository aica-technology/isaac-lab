from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import RigidObject
import torch


def reset_ball_in_spoon(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    spoon_cfg: SceneEntityCfg = SceneEntityCfg("spoon_frame"),
):
    # Get spoon frame object
    spoon_frame: FrameTransformer = env.scene[spoon_cfg.name]
    
    # Get ball object
    ball: RigidObject = env.scene[ball_cfg.name]

    # Get the position of the spoon frame (assuming [N, 1, 3])
    ball_positions = spoon_frame.data.target_pos_w[env_ids, 0, :]  # shape: [num_envs, 3]

    # Fixed upward orientation as a quaternion (w, x, y, z) = [1, 0, 0, 0]
    ball_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=ball.device).expand(len(env_ids), 4)

    # Combine position and orientation into pose
    ball_poses = torch.cat([ball_positions, ball_orientations], dim=-1)  # shape: [num_envs, 7]

    # Zero velocity: linear (3) + angular (3) = 6 DoF
    zero_velocities = torch.zeros((len(env_ids), 6), device=ball.device)

    # Write pose and velocity into simulation
    ball.write_root_pose_to_sim(ball_poses, env_ids=env_ids)
    ball.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)

