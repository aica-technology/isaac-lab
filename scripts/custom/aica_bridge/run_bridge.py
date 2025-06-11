from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Run the AICA bridge.")

parser.add_argument("--scene", type=str, help="Scene name to load.")
parser.add_argument("--rate", type=float, default=100.0, help="Simulation rate in Hz.")
parser.add_argument("--end_effector", type=str, default="wrist_3_link", help="End effector name.")
parser.add_argument("--force_sensor", type=str, default=None, help="Force sensor name.")
parser.add_argument("--ip_address", type=str, default="*", help="IP address of the AICA server.")
parser.add_argument("--state_port", type=int, default=1801, help="Port for the state socket.")
parser.add_argument("--command_port", type=int, default=1802, help="Port for the command socket.")
parser.add_argument("--force_port", type=int, default=1803, help="Port for the force sensor.")
parser.add_argument(
    "--command_interface", type=str, default="position", help="Command interface to use (default: position)."
)

AppLauncher.add_app_launcher_args(parser)

arguments = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=arguments.headless, device=arguments.device)
simulation_app = app_launcher.app

from typing import Optional
import argparse
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

import state_representation as sr
from scripts.custom.aica_bridge.scenes import scenes
from scripts.custom.aica_bridge.bridge.aica_bridge import AICABridge
from scripts.custom.aica_bridge.bridge.config_classes import BridgeConfig
import time


class Simulator:
    def __init__(
        self,
        bridge_config: BridgeConfig,
        simulation_config: SimulationCfg,
        scene_name: str,
        end_effector: str,
        command_interface: str = "position",
    ):
        """
        Initialize the simulator with the given configuration.
        Args:
            bridge_config (BridgeConfig): Configuration for the AICA Bridge.

            scene_name (str): Name of the scene to load.

            end_effector (str): Name of the end effector to control.

            command_interface (str): Command interface to use, either 'position' or 'velocity'.
        """
        self._sim = SimulationContext(simulation_config)
        self._sim.add_physics_callback("state_callback", self._state_callback)
        self._sim.add_physics_callback("command_callback", self._command_callback)

        self._command_interface = command_interface

        scene_cfg = scenes[scene_name](num_envs=1, env_spacing=2)
        self._scene = InteractiveScene(scene_cfg)
        self._robot: Articulation = self._scene["robot"]
        self._robot_joint_ids = SceneEntityCfg("robot", joint_names=[".*"], body_names=[end_effector]).joint_ids

        self._bridge = AICABridge(bridge_config, robot_joint_ids=self._robot_joint_ids)

    def run(self) -> None:
        """Initialize and run the simulation loop until the app is closed."""
        self._sim.reset()

        physics_dt = self._sim.get_physics_dt()
        render_dt = 1.0 / 60.0
        time_to_render = 0.0

        while simulation_app.is_running():
            start_time = time.time()
            if not self._bridge.is_active:
                self._bridge.activate(self._robot.data.joint_names)
                if self._bridge.is_active:
                    self._initialize_robot()
                    print("AICA Bridge activated. Waiting for commands...")
                else:
                    print("Failed to activate AICA Bridge...")
                    break

            self._sim.step(render=False)
            self._scene.update(physics_dt)

            if not arguments.headless:
                time_to_render += time.time() - start_time
                if time_to_render >= render_dt:
                    self._sim.render()
                    time_to_render = 0.0

            elapsed_time = time.time() - start_time
            sleep_duration = physics_dt - elapsed_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        simulation_app.close()

    def _state_callback(self, _):
        """
        Callback to send the current state of the robot to the AICA Bridge.
        """
        if self._bridge.is_active:
            self._bridge.send_states(self._robot, self._scene)
        else:
            print("AICA Bridge is not active. Cannot send states.")

    def _command_callback(self, _):
        """
        Callback to receive commands from the AICA Bridge and apply them to the robot.
        """
        if self._bridge.is_active:
            position_command, velocity_command = self._bridge.receive_commands()
            self._apply_command(position_command, velocity_command)
            self._scene.write_data_to_sim()
        else:
            print("AICA Bridge is not active. Cannot receive commands.")

    def _initialize_robot(self) -> None:
        """Set robot to default pose at startup. If using velocity control, set stiffness to zero."""
        defaults = self._robot.data
        self._robot.write_joint_state_to_sim(defaults.default_joint_pos, defaults.default_joint_vel)
        if self._command_interface == "velocity":
            # set the stiffness to zero for velocity control
            self._robot.write_joint_stiffness_to_sim(
                torch.zeros_like(defaults.default_joint_stiffness, device=self._robot.device)
            )
        self._robot.reset()

    def _apply_command(
        self,
        position_command: Optional[sr.JointState],
        velocity_command: Optional[sr.JointState],
    ) -> None:
        """
        Apply the received position or velocity command to the robot.

        Args:
            position_command: Joint position command, if any.
            velocity_command: Joint velocity command, if any.
        Raises:
            ValueError: If both position and velocity commands are received, or if the command interface does not match.
        """
        if self._command_interface == "position":
            if position_command:
                target = torch.tensor(position_command.get_positions(), device=self._robot.device)
                self._robot.set_joint_position_target(target, joint_ids=self._robot_joint_ids)
            elif velocity_command:
                raise ValueError(
                    "Received velocity command while using position interface. Use the velocity interface instead."
                )
            else:
                # maintain current state
                current = self._robot.data.joint_pos[0, self._robot_joint_ids]
                self._robot.set_joint_position_target(current, joint_ids=self._robot_joint_ids)
        elif self._command_interface == "velocity":
            if velocity_command:
                target = torch.tensor(velocity_command.get_velocities(), device=self._robot.device)
                self._robot.set_joint_velocity_target(target, joint_ids=self._robot_joint_ids)
            elif position_command:
                raise ValueError(
                    "Received position command while using velocity interface. Use the position interface instead."
                )
            else:
                self._robot.set_joint_velocity_target(
                    torch.zeros_like(self._robot.data.joint_vel[0, self._robot_joint_ids], device=self._robot.device),
                    joint_ids=self._robot_joint_ids,
                )


def main() -> None:
    # check if the scene name is valid
    if arguments.scene not in scenes:
        raise ValueError(f"Invalid scene name: {arguments.scene}. Available scenes are: {list(scenes.keys())}")

    if arguments.command_interface not in ["position", "velocity"]:
        raise ValueError(
            f"Invalid command interface: {arguments.command_interface}. Available options are: position, velocity"
        )

    if arguments.rate <= 0:
        raise ValueError(f"Invalid rate: {arguments.rate}. Rate must be a positive number.")

    bridge_config = BridgeConfig(
        address=arguments.ip_address,
        state_port=arguments.state_port,
        command_port=arguments.command_port,
        force_port=arguments.force_port,
        force_sensor_name=arguments.force_sensor,
    )

    sim = Simulator(
        bridge_config=bridge_config,
        scene_name=arguments.scene,
        simulation_config=SimulationCfg(
            dt=1.0 / arguments.rate,
            device=arguments.device,
        ),
        end_effector=arguments.end_effector,
        command_interface=arguments.command_interface,
    )
    sim.run()


if __name__ == "__main__":
    main()
