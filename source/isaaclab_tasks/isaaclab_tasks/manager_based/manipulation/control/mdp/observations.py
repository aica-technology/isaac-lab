import torch
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import transform_points
from isaaclab.envs import ManagerBasedRLEnv


def measured_forces(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    force_w, _ = torch.max(torch.mean(contact_sensor.data.force_matrix_w, dim=1), dim=1)  # type: ignore
    end_effector: FrameTransformer = env.scene[end_effector_cfg.name]
    ee_quat_w = end_effector.data.target_quat_w[..., 0, :]
    force_ee = transform_points(force_w.unsqueeze(1), quat=ee_quat_w)
    return force_ee.squeeze() if not force_ee.shape[0] == 1 else force_ee
