import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

def object_goal_distance_incentive(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    return torch.exp(-torch.norm(command - object.data.root_pos_w[:, :3], dim=1) / 5)


def object_goal_distance_fine_grained(
    env,
    std: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    return 1 - torch.tanh(torch.norm(command - object.data.root_pos_w[:, :3], dim=1)/std)