import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

def object_goal_distance_penalty(
    env,
    bin_cfg:  SceneEntityCfg = SceneEntityCfg("bin"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    ball: RigidObject = env.scene[ball_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]

    return torch.norm(bin.data.root_pos_w[:, :3] - ball.data.root_pos_w[:, :3], dim=1)


def object_goal_distance_fine_grained(
    env,
    std: float,
    bin_cfg:  SceneEntityCfg = SceneEntityCfg("bin"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    ball: RigidObject = env.scene[ball_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]
    
    return 1 - torch.tanh(torch.norm(bin.data.root_pos_w[:, :3] - ball.data.root_pos_w[:, :3], dim=1)/std)


def object_near_target(
    env,
    bin_cfg:  SceneEntityCfg = SceneEntityCfg("bin"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    ball: RigidObject = env.scene[ball_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]
    
    # offset tensor that also makes sure that the robot is getting rewards in the initial state
    enter_bin_offset = torch.tensor([0.3, 0.3, -0.3], device=bin.device)

    return torch.exp(-torch.norm(bin.data.root_pos_w[:, :3] - enter_bin_offset - ball.data.root_pos_w[:, :3], dim=1)*3) - 0.03
