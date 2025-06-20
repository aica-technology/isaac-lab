from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import (
        VelocityPIActuatorCfg,
    )

class VelocityPIActuator(ActuatorBase):
    cfg: VelocityPIActuatorCfg
    """Velocity PI actuator model."""
    def __init__(self, cfg: VelocityPIActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._error_accumulator = torch.zeros(
            (self._num_envs, self.num_joints), device=self._device, dtype=torch.float32
        )

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
        error_velocity = control_action.joint_velocities - joint_vel
        self._error_accumulator += error_velocity * self.cfg.delta_time
        
        # clip the error accumulator to prevent integral windup
        self._error_accumulator = torch.clamp(
            self._error_accumulator, -self.cfg.max_integral_error, self.cfg.max_integral_error
        )

        feed_forward_velocity = control_action.joint_velocities * self.cfg.delta_time  # multiply by dt

        self.computed_effort = (
            self.damping * error_velocity # k_d * error_velocity
            + self.stiffness * feed_forward_velocity # k_p * feed_forward_velocity
            + self._error_accumulator * self.cfg.integral_gain # k_i * integral of error_velocity
        )

        # calculate the desired joint torques
        self.computed_effort = self.damping * error_velocity + control_action.joint_efforts

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
