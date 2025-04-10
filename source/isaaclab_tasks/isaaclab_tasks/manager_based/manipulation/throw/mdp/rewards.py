import torch

from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    return torch.norm(command - object.data.root_pos_w[:, :3], dim=1)

def object_goal_distance_fine_grained(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    return 1 - torch.tanh(torch.norm(command - object.data.root_pos_w[:, :3], dim=1)/std)

def object_throwing_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
) -> torch.Tensor:
    
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2]