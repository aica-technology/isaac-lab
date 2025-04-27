from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .observations import setpoint_error, measured_forces
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
    stiffness: float = 50
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    desired_contact_force = command[:, 7:] # Nx3

    experienced_forces = measured_forces(env, contact_sensor_config, end_effector_config) # Nx3
    contact_force_error = torch.norm(desired_contact_force - experienced_forces, dim=1) # Nx1

    # FIXME: remove
    """
    with open("source/isaaclab_tasks/logs_reward.txt", "a") as file:
        exp_forces = experienced_forces.cpu().numpy()
        #position_err = position_error.cpu().numpy()
        desired_contact = desired_contact_force.cpu().numpy()
        file.write(
            f"{exp_forces[0][0]}, {exp_forces[0][1]}, {exp_forces[0][2]}, {desired_contact[0][0]}, {desired_contact[0][1]}, {desired_contact[0][2]}\n"
        )
    """
    return contact_force_error / stiffness

def state_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    stiffness: float = 50,
    damping: float = 5,
) -> torch.Tensor:
    return setpoint_error(env, command_name, stiffness=stiffness, damping=damping) / stiffness # Nx3


def force_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_config: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_config: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.2,
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


def force_limit_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    maximum_limit: float = 100
) -> torch.Tensor:

    force = torch.abs(measured_forces(env, contact_sensor_cfg, end_effector_cfg))

    mask = torch.max(force > maximum_limit * torch.ones_like(force), dim=1)[0]
    reward = torch.zeros_like(mask)
    reward[mask] = -1
    
    return reward


def in_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
):
    command = env.command_manager.get_command(command_name)
    end_effector: FrameTransformer = env.scene[end_effector_cfg.name]
    table: RigidObject = env.scene[asset_cfg.name]

    ee_pose = end_effector.data.target_pos_w[..., 0, :3]
    desired_z_position = command[:, 2]
    
    # Create a mask for values where desired_z_position is below the table height
    below_table_mask = desired_z_position <= table.data.root_pos_w[:, 2]

    # Initialize output tensor to be the same shape as desired_z_position
    reward = torch.ones_like(desired_z_position) 

    # Apply the 'if' block for elements below the table height
    reward[below_table_mask] = 1 - torch.tanh(torch.norm(ee_pose[below_table_mask] - command[:, :3][below_table_mask]))

    return reward
