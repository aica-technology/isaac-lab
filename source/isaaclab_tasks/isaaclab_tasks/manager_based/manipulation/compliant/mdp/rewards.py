# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .observations import impedance_law_desired_forces, measured_forces
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
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
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def force_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_config: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_config: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    desired_contact_force = command[:, 7:]

    virtual_forces = impedance_law_desired_forces(env, command_name)
    experienced_forces = measured_forces(env, contact_sensor_config, end_effector_config)

    scaling_factor = torch.exp(-torch.norm(experienced_forces, dim=1))

    maximum_force_experienced = torch.maximum(
        torch.max(torch.abs(experienced_forces), dim=1)[0],
        torch.ones(experienced_forces.shape[0], device=scaling_factor.device),
    )

    maximum_desired_contact = torch.maximum(
        torch.max(torch.abs(desired_contact_force), dim=1)[0],
        torch.zeros(desired_contact_force.shape[0], device=scaling_factor.device),
    )

    # minimize this error
    return scaling_factor * torch.norm(virtual_forces, dim=1) + (
        1 - scaling_factor
    ) * torch.norm(desired_contact_force - experienced_forces, dim=1) /(maximum_force_experienced + maximum_desired_contact)


def force_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_config: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_config: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 10,
) -> torch.Tensor:
    return 1 - torch.tanh(force_command_error(env, command_name, contact_sensor_config, end_effector_config) / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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


def action_termination(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize actions using L2 squared kernel with a distance-based scaling factor.
    The penalty reduces smoothly as the end-effector approaches the target.
    """
    # Calculate the L2 squared penalty for the action
    asset: Articulation = env.scene[asset_cfg.name]
    velocity =  torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

    # Compute the distance to the target
    position_error = position_command_error(env, command_name, asset_cfg)

    # Apply a scaling factor that diminishes as the robot approaches the target
    penalty = velocity * torch.exp(-10*position_error)
    return penalty


def maximum_measured_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    desired_contact_force = command[:, 7:]
    force = measured_forces(env, contact_sensor_cfg, end_effector_cfg)
    return torch.norm(torch.maximum(desired_contact_force - force, torch.zeros_like(force)), dim=1)

def in_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    table_height: float,
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
):
    command = env.command_manager.get_command(command_name)
    end_effector: FrameTransformer = env.scene[end_effector_cfg.name]
    ee_pose = end_effector.data.target_pos_w[..., 0, :3]
    desired_z_position = command[:, 2]

    # Create a mask for values where desired_z_position is below the table height
    below_table_mask = desired_z_position <= table_height

    # Initialize output tensor to be the same shape as desired_z_position
    reward = torch.ones_like(desired_z_position) 

    # Apply the 'if' block for elements below the table height
    reward[below_table_mask] = 1 - torch.tanh(torch.norm(ee_pose[below_table_mask] - command[:, :3][below_table_mask]))

    return reward
