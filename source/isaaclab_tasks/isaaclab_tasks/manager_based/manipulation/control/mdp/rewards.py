from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .observations import measured_forces

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

    experienced_forces = measured_forces(env, contact_sensor_config, end_effector_config)
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
        (1 / std) * setpoint_error(env, command_name, stiffness=stiffness, damping=damping) / stiffness
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

    force = torch.abs(measured_forces(env, contact_sensor_cfg, end_effector_cfg))

    mask = torch.max(force > maximum_limit * torch.ones_like(force), dim=1)[0]
    reward = torch.zeros_like(mask)
    reward[mask] = -1

    return reward
