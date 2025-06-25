from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import (
        VelocityPIDActuatorCfg,
    )


class VelocityPIDActuator(ActuatorBase):
    cfg: VelocityPIDActuatorCfg
    """Velocity PID actuator model."""

    def __init__(self, cfg: VelocityPIDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._error_integral = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)
        # last error
        self._last_error = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)

        self._proportional_gain = self._parse_joint_parameter(self.cfg.proportional_gain, None)
        self._derivate_gain = self._parse_joint_parameter(self.cfg.derivative_gain, None)
        self._integral_gain = self._parse_joint_parameter(self.cfg.integral_gain, None)
        self._max_integral_error = self._parse_joint_parameter(self.cfg.max_integral_error, None)

    def reset(self, env_ids: Sequence[int]):
        """Reset the actuator state."""
        self._error_integral[env_ids, :] = 0.0
        self._last_error[env_ids, :] = 0.0

    def set_propertional_gain(self, proportional_gain: torch.Tensor):
        """Set the proportional gain."""
        self._proportional_gain = proportional_gain

    def set_derivative_gain(self, derivative_gain: torch.Tensor):
        """Set the derivative gain."""
        self._derivate_gain = derivative_gain

    def set_integral_gain(self, integral_gain: torch.Tensor):
        """Set the integral gain."""
        self._integral_gain = integral_gain

    def compute(
        self, control_action: ArticulationActions, joint_vel: torch.Tensor, mass_matrix: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
        error_velocity = control_action.joint_velocities - joint_vel
        self._error_integral += error_velocity * self.cfg.delta_time
        self._error_derivative = (error_velocity - self._last_error) / self.cfg.delta_time

        self._error_integral = torch.clamp(
            self._error_integral, self._max_integral_error * -1, self._max_integral_error
        )

        self._last_error = error_velocity.clone()

        self.computed_effort = (
            self._proportional_gain * error_velocity  # kp * e_v
            + self._integral_gain * self._error_integral  # ki * e_i
            + self._derivate_gain * self._error_derivative  # kd * e_d
        )
        # apply mass matrix
        self.computed_effort = torch.bmm(mass_matrix, self.computed_effort.unsqueeze(2)).squeeze(2)
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
