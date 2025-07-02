from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Run the AICA bridge.")

parser.add_argument("--scene", type=str, help="Scene name to load.")
parser.add_argument("--rate", type=float, default=100.0, help="Simulation rate in Hz.")
parser.add_argument("--force_sensor", type=str, default=None, help="Force sensor name.")
parser.add_argument("--ip_address", type=str, default="*", help="IP address of the AICA server.")
parser.add_argument("--state_port", type=int, default=1801, help="Port for the state socket.")
parser.add_argument("--command_port", type=int, default=1802, help="Port for the command socket.")
parser.add_argument("--force_port", type=int, default=1803, help="Port for the force sensor.")
parser.add_argument("--joint_names", nargs="+", help="List of joint names to control.")
parser.add_argument(
    "--command_interface", type=str, default="positions", help="Command interface to use (default: positions)."
)

AppLauncher.add_app_launcher_args(parser)

arguments = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=arguments.headless, device=arguments.device)
simulation_app = app_launcher.app


import argparse
import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

import state_representation as sr
from state_representation import StateType
from scripts.custom.aica_bridge.scenes import scenes
from scripts.custom.aica_bridge.bridge.aica_bridge import AICABridge
from scripts.custom.aica_bridge.bridge.config_classes import BridgeConfig
import time

STATE_TYPE_TO_STRING = {
    StateType.JOINT_POSITIONS: "positions",
    StateType.JOINT_VELOCITIES: "velocities",
    StateType.JOINT_TORQUES: "torques",
}

class Simulator:
    def __init__(
        self,
        bridge_config: BridgeConfig,
        simulation_config: SimulationCfg,
        scene_name: str,
        request_joint_names: list[str],
        command_interface: str = "positions",
    ):
        """
        Initialize the simulator with the given configuration.
        Args:
            bridge_config (BridgeConfig): Configuration for the AICA Bridge.

            scene_name (str): Name of the scene to load.

            command_interface (str): Command interface to use, either 'positions', 'velocities', or 'torques'.
        """
        self._sim = SimulationContext(simulation_config)
        self._sim.add_physics_callback("state_callback", self._state_callback)
        self._sim.add_physics_callback("command_callback", self._command_callback)

        self._command_interface = command_interface

        scene_cfg = scenes[scene_name](num_envs=1, env_spacing=2)
        self._scene = InteractiveScene(scene_cfg)

        self._robot: Articulation = self._scene["robot"]
        self._request_joint_names = request_joint_names
        self._robot_joint_ids = None

        self._bridge = AICABridge(bridge_config)

    def run(self) -> None:
        """Initialize and run the simulation loop until the app is closed."""
        self._sim.reset()

        physics_dt = self._sim.get_physics_dt()
        render_dt = 1.0 / 60.0
        time_to_render = 0.0

        while simulation_app.is_running():
            start_time = time.time()
            if not self._bridge.is_active:
                self._robot_joint_ids = self._robot.find_joints(self._request_joint_names)[0]
                self._bridge.activate(
                    [self._robot.data.joint_names[joint_id] for joint_id in self._robot_joint_ids],
                    self._robot_joint_ids,
                )
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
            command = self._bridge.receive_commands()
            self._apply_command(command)
            self._scene.write_data_to_sim()
        else:
            print("AICA Bridge is not active. Cannot receive commands.")

    def _initialize_robot(self) -> None:
        """Set robot to default pose at startup. If using velocities control, set stiffness to zero."""
        defaults = self._robot.data
        self._robot.write_joint_state_to_sim(defaults.default_joint_pos, defaults.default_joint_vel)
        if self._command_interface == "velocities":
            # set the stiffness to zero for velocities control
            self._robot.write_joint_stiffness_to_sim(
                torch.zeros_like(defaults.default_joint_stiffness, device=self._robot.device)
            )
        self._robot.reset()

    def _apply_command(
        self,
        command: sr.JointState,
    ) -> None:
        """
        Apply the received positions or velocities command to the robot.

        Args:
            command (sr.JointState): The command containing the joint state
        Raises:
            ValueError: If the command type does not match the expected type based on the command interface.
        """
        command_translators = {
            "positions": {
                "command_getter": command.get_positions if command is not None else lambda: None,
                "sim_setter": self._robot.set_joint_position_target,
                "no_command_state": self._robot.data.joint_pos[0, self._robot_joint_ids],
            },
            "velocities": {
                "command_getter": command.get_velocities if command is not None else lambda: None,
                "sim_setter": self._robot.set_joint_velocity_target,
                "no_command_state": torch.zeros_like(
                    self._robot.data.joint_vel[0, self._robot_joint_ids], device=self._robot.device
                ),
            },
            "torques": {
                "command_getter": command.get_torques if command is not None else lambda: None,
                "sim_setter": self._robot.set_joint_effort_target,
                "no_command_state": torch.zeros_like(
                    self._robot.data.joint_vel[0, self._robot_joint_ids], device=self._robot.device
                ),
            },
        }

        if command is not None:
            if STATE_TYPE_TO_STRING[command.get_type()] != self._command_interface:
                raise ValueError(
                    f"Received a command of type {STATE_TYPE_TO_STRING[command.get_type()]}, but the command interface is set to {self._command_interface}."
                )

            target = torch.tensor(
                command_translators[self._command_interface]["command_getter"](), device=self._robot.device
            ).to(dtype=torch.float32)
            command_translators[self._command_interface]["sim_setter"](target, joint_ids=self._robot_joint_ids)
        else:
            command_translators[self._command_interface]["sim_setter"](
                command_translators[self._command_interface]["no_command_state"], joint_ids=self._robot_joint_ids
            )


def main() -> None:
    # check if the scene name is valid
    if arguments.scene not in scenes:
        raise ValueError(f"Invalid scene name: {arguments.scene}. Available scenes are: {list(scenes.keys())}")

    if arguments.command_interface not in ["positions", "velocities", "torques"]:
        raise ValueError(
            f"Invalid command interface: {arguments.command_interface}. Available options are: positions, velocities, torques"
        )

    if arguments.joint_names is None:
        raise ValueError("Joint names must be provided. Use --joint_names to specify them.")

    else:
        if len(arguments.joint_names) == 0:
            raise ValueError("Joint names list cannot be empty. Provide at least one joint name.")

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
        request_joint_names=arguments.joint_names,
        command_interface=arguments.command_interface,
    )
    sim.run()


if __name__ == "__main__":
    main()
