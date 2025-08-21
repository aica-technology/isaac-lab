from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .observations import measured_forces_in_ee_frame, measured_forces_in_world_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, error_type="position"
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    if error_type == "position":
        des_pos_b = command[:, :3]
        desired_setpoint, _ = combine_frame_transforms(
            asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
        )
        current_setpoint = asset.data.body_state_w[:, 6, :3]
    else:
        des_vel_b = torch.zeros_like(command[:, :3], device=asset.device)
        desired_setpoint, _ = combine_frame_transforms(
            asset.data.root_state_w[:, 7:10], asset.data.root_state_w[:, 3:7], des_vel_b
        )
        current_setpoint = asset.data.body_state_w[:, 6, 7:10]
    return current_setpoint - desired_setpoint


def setpoint_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stiffness: float = 50,
    damping: float = 20,
) -> torch.Tensor:
    position_error = command_error(env, command_name, robot_cfg, error_type="position")
    velocity_error = command_error(env, command_name, robot_cfg, error_type="velocity")
    return stiffness * torch.norm(position_error, dim=1) + damping * torch.norm(velocity_error, dim=1)


def force_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_config: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    end_effector_config: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    stiffness: float = 50,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    desired_contact_force = command[:, 7:]

    experienced_forces = measured_forces_in_ee_frame(env, contact_sensor_config, end_effector_config)
    contact_force_error = torch.norm(desired_contact_force - experienced_forces, dim=1)
    return contact_force_error / stiffness


def state_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    stiffness: float = 50,
    damping: float = 5,
) -> torch.Tensor:
    return setpoint_error(env, command_name, stiffness=stiffness, damping=damping) / stiffness


def state_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    stiffness: float = 50,
    damping: float = 5,
    std: float = 0.1,
) -> torch.Tensor:
    return 1 - torch.tanh(
        setpoint_error(env, command_name, stiffness=stiffness, damping=damping) / (stiffness * std)
    )


def force_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_config: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    end_effector_config: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.2,
) -> torch.Tensor:
    return 1 - torch.tanh(force_command_error(env, command_name, contact_sensor_config, end_effector_config) / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def force_limit_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    maximum_limit: float = 100,
) -> torch.Tensor:

    force = torch.abs(measured_forces_in_ee_frame(env, contact_sensor_cfg, end_effector_cfg))

    mask = torch.max(force > maximum_limit * torch.ones_like(force), dim=1)[0]
    reward = torch.zeros_like(mask)
    reward[mask] = -1
    return reward

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, raw_vector=False) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore

    if raw_vector:
        return curr_pos_w - des_pos_w
    
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def action_termination(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize actions using L2 squared kernel with a distance-based scaling factor.
    The penalty reduces smoothly as the end-effector approaches the target.
    """
    # Calculate the L2 squared penalty for the action
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    
    # Compute the distance to the target
    position_error = position_command_error(env, command_name, asset_cfg)
    
    # Apply a scaling factor that diminishes as the robot approaches the target
    penalty = velocity * torch.exp(-position_error)
    
    return penalty

def velocity_contact(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    limit: float = 100,
    scale: float = 0.06,
) -> torch.Tensor:
    """Compute the velocity of the contact sensor."""
    force = torch.norm(measured_forces_in_ee_frame(env, contact_sensor_cfg, end_effector_cfg), dim=1) / limit
    force = torch.clamp(force, min=0, max=2.0)
    velocity = torch.norm(scale * env.action_manager.action[:, :3], dim=1)
    return velocity * torch.exp(force)

def force_direction_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    command_name: str = "ee_pose",
    limit: float = 1,
) -> torch.Tensor:
    """Reward the force in the direction of the position error."""
    force = measured_forces_in_world_frame(env, contact_sensor_cfg)
    position_error = position_command_error(env, command_name, asset_cfg, raw_vector=True)

    force_position = torch.sum(force * position_error, dim=1)
    force_position = torch.clamp(force_position, -limit, limit)

    return force_position

def force_tracking_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    command_name: str = "ee_pose",
    stiffness = 120
) -> torch.Tensor:
    """Compute the force tracking penalty."""
    force_world = measured_forces_in_world_frame(env, contact_sensor_cfg)
    position_error = position_command_error(env, command_name, asset_cfg, raw_vector=True)

    f_target = position_error * stiffness
    force_error = torch.norm(force_world - f_target, dim=1)
    return force_error
    