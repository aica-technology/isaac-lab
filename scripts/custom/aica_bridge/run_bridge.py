from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from typing import Optional, Union
import argparse
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

import state_representation as sr
from scripts.custom.aica_bridge.scenes import scenes
from scripts.custom.aica_bridge.bridge.aica_bridge import AICABridge
from scripts.custom.aica_bridge.bridge.config_classes import ZMQConfig, SimulationParameters

class Simulator:
    def __init__(
        self,
        zmq_config: ZMQConfig,
        scene_name: str,
        end_effector: str,
        force_sensor_name: Union[str, None] = None,
        command_interface: str = "position",
    ):
        sim_cfg = SimulationCfg(device=SimulationParameters().device, dt=SimulationParameters().dt)
        self._sim = SimulationContext(sim_cfg)
        self._command_interface = command_interface

        scene_cfg = scenes[scene_name](num_envs=1, env_spacing=2)
        self._scene = InteractiveScene(scene_cfg)
        self._robot: Articulation = self._scene["robot"]
        self._robot_joint_ids = SceneEntityCfg("robot", joint_names=[".*"], body_names=[end_effector]).joint_ids
        
        self._bridge = AICABridge(zmq_config, robot_joint_ids=self._robot_joint_ids, force_sensor_name=force_sensor_name)
        self._setup_done = False

    def run(self) -> None:
        """Initialize and run the simulation loop until the app is closed."""
        self._sim.reset()

        self._bridge.open()

        physics_dt = self._sim.get_physics_dt()
        while simulation_app.is_running():
            if not self._setup_done:
                self._initialize_robot()
                self._bridge.set_joint_state_names(self._robot.data.joint_names)
                self._setup_done = True
            else:
                self._bridge.send_states(self._robot, self._scene)
                position_command, velocity_command = self._bridge.receive_commands()
                self._apply_command(position_command, velocity_command)

            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(physics_dt)

        simulation_app.close()


    def _initialize_robot(self) -> None:
        """Set robot to default pose at startup."""
        defaults = self._robot.data
        self._robot.write_joint_state_to_sim(defaults.default_joint_pos, defaults.default_joint_vel)
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
        """
        if self._command_interface == "position":
            if position_command:
                target = torch.tensor(position_command.get_positions(), device=self._robot.device)
                self._robot.set_joint_position_target(target, joint_ids=self._robot_joint_ids)
            else:
                # maintain current state
                current = self._robot.data.joint_pos[0, self._robot_joint_ids]
                self._robot.set_joint_position_target(current, joint_ids=self._robot_joint_ids)
        elif self._command_interface == "velocity":
            if velocity_command:
                target = torch.tensor(velocity_command.get_velocities(), device=self._robot.device)
                self._robot.set_joint_velocity_target(target, joint_ids=self._robot_joint_ids)
            else:
                self._robot.set_joint_velocity_target(
                    torch.zeros_like(self._robot.data.joint_vel[0, self._robot_joint_ids], device=self._robot.device),
                    joint_ids=self._robot_joint_ids,
                )
                


def main() -> None:
    # create a argparse that takes as input the scene name and the end effector name
    parser = argparse.ArgumentParser(description="Run the AICA bridge.")
    parser.add_argument("--scene", type=str, help="Scene name to load.")
    parser.add_argument("--end_effector", type=str, default="wrist_3_link", help="End effector name.")
    parser.add_argument("--force_sensor", type=str, default=None, help="Force sensor name.")
    parser.add_argument("--ip_address", type=str, default="*", help="IP address of the AICA server.")
    parser.add_argument("--state_port", type=int, default="1801", help="Port for the state socket.")
    parser.add_argument("--command_port", type=str, default="1802", help="Port for the command socket.")
    parser.add_argument("--force_port", type=str, default="1803", help="Port for the force sensor.")
    parser.add_argument("--command_interface", type=str, default="position", help="Command interface to use (default: position).")
    
    # parse the arguments
    arguments = parser.parse_args()
    
    # check if the scene name is valid
    if arguments.scene not in scenes:
        raise ValueError(f"Invalid scene name: {arguments.scene}. Available scenes are: {list(scenes.keys())}")

    if arguments.command_interface not in ["position", "velocity"]:
        raise ValueError(f"Invalid command interface: {arguments.command_interface}. Available options are: position, velocity")

    # create zmq config
    zmq_cfg = ZMQConfig(
        address=arguments.ip_address,
        state_port=arguments.state_port,
        command_port=arguments.command_port,
        force_port=arguments.force_port,
    )

    sim = Simulator(    
        zmq_config=zmq_cfg,
        scene_name=arguments.scene,
        end_effector=arguments.end_effector,
        force_sensor_name=arguments.force_sensor,
        command_interface=arguments.command_interface,
    )
    sim.run()


if __name__ == "__main__":
    main()