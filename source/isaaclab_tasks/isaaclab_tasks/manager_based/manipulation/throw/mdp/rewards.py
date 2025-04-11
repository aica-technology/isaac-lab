import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

def position_command_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position and velocity using the tanh kernel.
    """
    
    # extract the asset (to enable type hinting)
    distance = position_command_error(env, command_name, asset_cfg)
    return 1 - torch.tanh(distance / std)

def velocity_command_reward(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error and velocity error in the vicinity of the taget using weight L2-norm.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    curr_velocity = asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10]  # type: ignore
    velocity_magnitude = torch.norm(curr_velocity, dim = 1)

    return torch.exp(-1 * torch.norm(curr_pos_w - des_pos_w, dim=1)) * torch.minimum(velocity_magnitude, torch.ones_like(velocity_magnitude, device=curr_velocity.device))

def orientation_command_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def object_distance_from_robot(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),    
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    rewards = torch.zeros_like(object.data.root_pos_w[:, 0]) 
    mask = object.data.root_pos_w[:, 0] > 2.0
    rewards[mask] = (torch.norm(object.data.root_pos_w[:, :3], dim=1)*object.data.root_pos_w[:, 2])[mask]
    return rewards


def object_goal_distance_penalty(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.norm(command - object.data.root_pos_w[:, :3], dim=1)


def object_goal_distance_fine_grained(
    env,
    std: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    return 1 - torch.tanh(torch.norm(command - object.data.root_pos_w[:, :3], dim=1)/std)


def object_near_target(
    env,
    minimum_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    mask = (object.data.root_pos_w[:, 2] < minimum_height) & (object.data.root_pos_w[:, 0] > 2.0)
    rewards = torch.zeros_like(object.data.root_pos_w[:, 2])
    rewards[mask] = (1 / (torch.norm(command - object.data.root_pos_w[:, :3], dim=1) + 0.1))[mask]
    return rewards

