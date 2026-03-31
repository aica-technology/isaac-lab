from dataclasses import MISSING
import logging

from source.isaaclab.isaaclab.utils import configclass

from . import actuator_pid
from .actuator_base_cfg import ActuatorBaseCfg

# import logger
logger = logging.getLogger(__name__)


"""
Explicit Actuator Models.
"""


@configclass
class VelocityPIDActuatorCfg(ActuatorBaseCfg):
    """Configuration for a PID actuator."""

    class_type: type = actuator_pid.VelocityPIDActuator

    stiffness = None
    """Stiffness gain of the PID controller. Defaults to None."""

    damping = None
    """Damping gain of the PID controller. Defaults to None."""

    integral_gain: dict[str, float] | float | None = MISSING
    """Integral gain of the PID controller."""

    derivative_gain: dict[str, float] | float | None = MISSING
    """Derivative gain of the PID controller."""

    proportional_gain: dict[str, float] | float | None = MISSING
    """Proportional gain of the PID controller."""

    max_integral_error: dict[str, float] | float | None = MISSING
    """Maximum integral error of the PID controller."""

    delta_time: float = MISSING
    """Time step of the simulation in seconds."""
