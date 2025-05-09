from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from typing import Optional, Union

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
        num_envs: int = 1,
        spacing: float = 2.0
    ):
        sim_cfg = SimulationCfg(device=SimulationParameters().device, dt=SimulationParameters().dt)
        self._sim = SimulationContext(sim_cfg)

        scene_cfg = scenes[scene_name](num_envs=num_envs, env_spacing=spacing)
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
                pos_cmd, vel_cmd = self._bridge.receive_commands()
                self._apply_command(pos_cmd, vel_cmd)

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
        pos_cmd: Optional[sr.JointState],
        vel_cmd: Optional[sr.JointState],
    ) -> None:
        """
        Apply the received position or velocity command to the robot.

        Args:
            pos_cmd: Joint position command, if any.
            vel_cmd: Joint velocity command, if any.
        """
        if pos_cmd:
            target = torch.tensor(pos_cmd.get_positions(), device=self._robot.device)
            self._robot.set_joint_position_target(target, joint_ids=self._robot_joint_ids)
        elif vel_cmd:
            raw_vel = torch.tensor(vel_cmd.get_velocities(), device=self._robot.device)
            # scale hack to match simulation range
            scaled = raw_vel * (1.0 / 60.0) + self._robot.data.default_joint_pos
            self._robot.set_joint_position_target(scaled, joint_ids=self._robot_joint_ids)
        else:
            # maintain current state
            current = self._robot.data.joint_pos[0, self._robot_joint_ids]
            self._robot.set_joint_position_target(current, joint_ids=self._robot_joint_ids)


def main() -> None:
    zmq_cfg = ZMQConfig()
    sim = Simulator(
        zmq_config=zmq_cfg,
        scene_name="lift_scene",
        end_effector="wrist_3_link",
        #force_sensor_name="ur_tcp_fts_sensor",
        num_envs=1,
        spacing=2.0,
    )
    sim.run()


if __name__ == "__main__":
    main()