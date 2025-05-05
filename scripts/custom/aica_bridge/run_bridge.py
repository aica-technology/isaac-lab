from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

## ZMQ

# AICA State Representation
import state_representation as sr
import clproto
from communication_interfaces.sockets import ZMQCombinedSocketsConfiguration, ZMQPublisherSubscriber, ZMQContext
from state_representation import JointPositions, JointVelocities, StateType
##
# Pre-defined configs
##
from isaaclab_assets import UR5E_CFG_IK

IP_ADDRESS = "0.0.0.0"
STATE_PORT = 1801
COMMAND_PORT = 1802

@configclass
class AICABridgeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    robot: ArticulationCfg = UR5E_CFG_IK.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(1.25, 1.0, 0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, max_depenetration_velocity=0.1),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.15), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=2,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TiltedWall"],
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot: Articulation = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["wrist_3_link"])

    context = ZMQContext(1)
    config = ZMQCombinedSocketsConfiguration(context, '*', '1801', '1802')
    sockets = ZMQPublisherSubscriber(config)
    sockets.open()

    sim_dt = sim.get_physics_dt()
    setup = False
    joint_state = None
    position_command = None
    velocity_command = None

    while simulation_app.is_running():
        if not setup:
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            joint_names = ["ur_" + joint_name for joint_name in robot.joint_names]
            setup = True
        else:
            joint_positions = robot.data.joint_pos[:, robot_entity_cfg.joint_ids][0, :]
            joint_velocities = robot.data.joint_vel[:, robot_entity_cfg.joint_ids][0, :]
            
            joint_state = sr.JointState("ur5e", joint_names)
            joint_state.set_positions(joint_positions.cpu().numpy())
            joint_state.set_velocities(joint_velocities.cpu().numpy())

            command = sockets.receive_bytes()
            if command:
                command = clproto.decode(command)
                if command.get_type() == StateType.JOINT_POSITIONS:
                    position_command = command
                    velocity_command = None
                elif command.get_type() == StateType.JOINT_VELOCITIES:
                    position_command = None
                    velocity_command = command



        if position_command:
            robot.set_joint_position_target(torch.tensor(position_command.get_positions()), joint_ids=robot_entity_cfg.joint_ids)
        elif velocity_command:
            robot.set_joint_velocity_target(velocity_command.get_velocities(), joint_ids=robot_entity_cfg.joint_ids)
        else:
            # if no command is received, use the default joint position
            robot.set_joint_position_target(joint_pos[:, robot_entity_cfg.joint_ids].clone(), joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()
        
        sim.step()

        scene.update(sim_dt)

        if joint_state:
            sockets.send_bytes(clproto.encode(joint_state, clproto.MessageType.JOINT_STATE_MESSAGE))


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")

    sim = SimulationContext(sim_cfg)

    scene_cfg = AICABridgeSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()