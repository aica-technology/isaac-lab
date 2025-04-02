import torch
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, transform_points
from isaaclab.envs import ManagerBasedRLEnv


def command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, error_type = "position") -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    if error_type == "position":
        des_pos_b = command[:, :3]
        desired_setpoint, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
        current_setpoint = asset.data.body_state_w[:, 6, :3]  # type: ignore
    else:
        des_vel_b = torch.zeros_like(command[:, :3], device=asset.device)
        desired_setpoint, _ = combine_frame_transforms(asset.data.root_state_w[:, 7:10], asset.data.root_state_w[:, 3:7], des_vel_b)
        current_setpoint = asset.data.body_state_w[:, 6, 7:10]  # type: ignore
    return current_setpoint - desired_setpoint

def setpoint_error(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stiffness: float = 300, damping: float = 10, use_velocity: bool = False) -> torch.Tensor:
    position_error = command_error(env, command_name, robot_cfg, error_type="position")
    if use_velocity:
        velocity_error = command_error(env, command_name, robot_cfg, error_type="velocity")
        return position_error + damping/stiffness * velocity_error
    else:
        return position_error


def measured_forces(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    force_w, _ = torch.max(torch.mean(contact_sensor.data.net_forces_w_history, dim=1), dim=1) # type: ignore
    end_effector: FrameTransformer = env.scene[end_effector_cfg.name]
    ee_quat_w = end_effector.data.target_quat_w[..., 0, :]
    force_ee = transform_points(
        force_w.unsqueeze(1), quat=ee_quat_w
    )
    return force_ee.squeeze() if not force_ee.shape[0] == 1 else force_ee

def measured_force_gradient(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    force_w = contact_sensor.data.net_forces_w_history
    force_gradient = torch.gradient(force_w, dim=1)[0] # type: ignore

    return  torch.max(torch.norm(force_gradient, dim=-1).squeeze(), dim=1)[0]


def desired_contact_force(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return command[:, 7:]
