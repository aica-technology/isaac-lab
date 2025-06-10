from typing import Optional, Union, List

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from communication_interfaces.sockets import (
    ZMQCombinedSocketsConfiguration,
    ZMQContext,
    ZMQPublisher,
    ZMQPublisherSubscriber,
    ZMQSocketConfiguration,
)
import clproto
import state_representation as sr
from state_representation import StateType
from scripts.custom.aica_bridge.bridge.utils import measured_forces
from scripts.custom.aica_bridge.bridge.config_classes import BridgeConfig


class AICABridge:
    def __init__(self, config: BridgeConfig, robot_joint_ids: List[int] | slice):
        """
        Initialize the AICA Bridge with the given configuration and robot joint IDs.

        Args:
            config (BridgeConfig): Configuration for the AICA Bridge which includes address, ports, and force
                sensor name.

            robot_joint_ids (List[int] | slice): Joint IDs of the robot to be controlled which are retrieved from the
                SceneEntityCfg of the robot.
        """

        self._context = ZMQContext(1)

        combined_cfg = ZMQCombinedSocketsConfiguration(
            self._context, config.address, str(config.state_port), str(config.command_port)
        )
        self._state_command_publisher_subscriber = ZMQPublisherSubscriber(combined_cfg)

        self._robot_joint_ids = robot_joint_ids
        self._joint_state: sr.JointState = None

        self._use_force_sensor = config.force_sensor_name is not None
        if self._use_force_sensor:
            self._cartesian_wrench = sr.CartesianWrench().Zero(config.force_sensor_name)
            self._force_publisher = ZMQPublisher(
                ZMQSocketConfiguration(self._context, config.address, str(config.force_port), True)
            )

        self.__is_active = False

    @property
    def is_active(self) -> bool:
        """Check if the AICA Bridge is active."""
        return self.__is_active

    def activate(self, joint_names: list[str]) -> None:
        """Open ZMQ sockets for bidirectional communication and initialize joint state."""
        if not self.__is_active:
            if self._use_force_sensor:
                self._force_publisher.open()

            self._state_command_publisher_subscriber.open()
            self._joint_state = sr.JointState("robot", joint_names)
            self.__is_active = True

    def receive_commands(self) -> tuple[Optional[sr.JointState], Optional[sr.JointState]]:
        """
        Receive and decode commands from AICA.

        Returns:
            A tuple of (position_command, velocity_command).
        """
        if self.__is_active:
            data = self._state_command_publisher_subscriber.receive_bytes()
            if not data:
                return None, None

            message = clproto.decode(data)
            if message.get_type() == StateType.JOINT_POSITIONS:
                return message, None
            if message.get_type() == StateType.JOINT_VELOCITIES:
                return None, message
            return None, None
        else:
            raise RuntimeError("AICA Bridge is not active. Please activate it before receiving commands.")

    def send_states(self, robot: Articulation, scene: InteractiveScene) -> None:
        """
        Publish current joint states and measured forces.
        """
        if self.__is_active:
            positions = robot.data.joint_pos[0, self._robot_joint_ids]
            velocities = robot.data.joint_vel[0, self._robot_joint_ids]
            self._joint_state.set_positions(positions.cpu().numpy())
            self._joint_state.set_velocities(velocities.cpu().numpy())

            self._state_command_publisher_subscriber.send_bytes(
                clproto.encode(self._joint_state, clproto.MessageType.JOINT_STATE_MESSAGE)
            )

            if self._use_force_sensor:
                forces = measured_forces(scene)[0][0].cpu().numpy()
                self._cartesian_wrench.set_force(forces)
                self._cartesian_wrench.set_torque(torch.zeros(3).cpu().numpy())

                self._force_publisher.send_bytes(
                    clproto.encode(self._cartesian_wrench, clproto.MessageType.CARTESIAN_WRENCH_MESSAGE)
                )
        else:
            raise RuntimeError("AICA Bridge is not active. Please activate it before sending states.")
