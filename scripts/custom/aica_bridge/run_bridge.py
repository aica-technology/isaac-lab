import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the AICA bridge.")

parser.add_argument("--scene", type=str, help="Scene name to load.")
parser.add_argument("--rate", type=float, default=100.0, help="Simulation rate in Hz.")
parser.add_argument("--ft_sensor_name", type=str, default=None, help="Force sensor name.")
parser.add_argument("--ip_address", type=str, default="*", help="IP address of the AICA server.")
parser.add_argument("--state_port", type=int, default=1801, help="Port for the state socket.")
parser.add_argument("--command_port", type=int, default=1802, help="Port for the command socket.")
parser.add_argument("--ft_sensor_port", type=int, default=1803, help="Port for the force sensor.")
parser.add_argument(
    "--joint_names", nargs="+", default=".*", help="List of joint names to control. Defaults to '.*' (all joints)."
)
parser.add_argument(
    "--command_interface", type=str, default="positions", help="Command interface to use (default: positions)."
)

AppLauncher.add_app_launcher_args(parser)

arguments = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=arguments.headless, device=arguments.device)
simulation_app = app_launcher.app

from scripts.custom.aica_bridge.bridge.simulator import Simulator
from scripts.custom.aica_bridge.bridge.config_classes import BridgeConfig
from scripts.custom.aica_bridge.scenes import scenes
from isaaclab.sim import SimulationCfg

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
        ft_sensor_port=arguments.ft_sensor_port,
        ft_sensor_name=arguments.ft_sensor_name,
    )

    sim = Simulator(
        simulation_app=simulation_app,
        bridge_config=bridge_config,
        scene_name=arguments.scene,
        simulation_config=SimulationCfg(
            dt=1.0 / arguments.rate,
            device=arguments.device,
        ),
        request_joint_names=arguments.joint_names,
        command_interface=arguments.command_interface,
        headless=arguments.headless,
    )
    sim.run()


if __name__ == "__main__":
    main()
