from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation
from isaaclab.sensors import ContactSensor, FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.utils.math import transform_points
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG  
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
## ZMQ

# AICA State Representation
import state_representation as sr
import clproto
from communication_interfaces.sockets import (
    ZMQCombinedSocketsConfiguration,
    ZMQPublisherSubscriber,
    ZMQContext,
    ZMQPublisher,
    ZMQSocketConfiguration,
)
from state_representation import StateType
##
# Pre-defined configs
##
from isaaclab_assets import UR5E_CFG_LOW_LEVEL

IP_ADDRESS = "0.0.0.0"
STATE_PORT = 1801
COMMAND_PORT = 1802
FORCE_PORT = 1803

ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"


def measured_forces(
    scene: InteractiveScene,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = scene[contact_sensor_cfg.name]
    force_w, _ = torch.max(torch.mean(contact_sensor.data.force_matrix_w, dim=1), dim=1) # type: ignore
    end_effector: FrameTransformer = scene[end_effector_cfg.name]
    ee_quat_w = end_effector.data.target_quat_w[..., 0, :]
    force_ee = transform_points(
        force_w.unsqueeze(1), quat=ee_quat_w
    )
    return force_ee

@configclass
class AICABridgeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    robot: ArticulationCfg = UR5E_CFG_LOW_LEVEL.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=ee_frame_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0, 0, 0),
                    ),
                ),
            ],
        )
    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot: Articulation = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["wrist_3_link"])

    context = ZMQContext(1)
    config = ZMQCombinedSocketsConfiguration(context, '*', '1801', '1802')
    state_command_sockets = ZMQPublisherSubscriber(config)
    force_state_sockets =  ZMQPublisher(ZMQSocketConfiguration(context, "*", "1803", True))
    state_command_sockets.open()
    force_state_sockets.open()

    sim_dt = sim.get_physics_dt()
    setup = False
    joint_state = None
    cartesian_wrench = sr.CartesianWrench().Zero("ur_tcp_fts_sensor")
    position_command = None
    velocity_command = None

    while simulation_app.is_running():
        if not setup:
            joint_positions = robot.data.default_joint_pos.clone()
            joint_velocities = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_positions, joint_velocities)
            robot.reset()
            joint_names = ["ur_" + joint_name for joint_name in robot.joint_names]
            setup = True
        else:
            joint_positions = robot.data.joint_pos[:, robot_entity_cfg.joint_ids][0, :]
            joint_velocities = robot.data.joint_vel[:, robot_entity_cfg.joint_ids][0, :]
            
            joint_state = sr.JointState("ur5e", joint_names)
            joint_state.set_positions(joint_positions.cpu().numpy())
            joint_state.set_velocities(joint_velocities.cpu().numpy())

            cartesian_wrench.set_force(
                measured_forces(scene)[0][0].cpu().numpy()
            )

            # set torque to zero
            cartesian_wrench.set_torque(torch.zeros(3).cpu().numpy())

            command = state_command_sockets.receive_bytes()
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
            robot.write_joint_stiffness_to_sim(1e5*torch.ones_like(joint_positions), joint_ids=robot_entity_cfg.joint_ids)
        elif velocity_command:
            robot.write_joint_stiffness_to_sim(torch.zeros_like(joint_positions), joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_velocity_target(torch.tensor(velocity_command.get_velocities()), joint_ids=robot_entity_cfg.joint_ids)
        else:
            robot.set_joint_position_target(joint_positions.clone(), joint_ids=robot_entity_cfg.joint_ids)
            robot.write_joint_stiffness_to_sim(1e5*torch.ones_like(joint_positions), joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()
        
        sim.step()

        scene.update(sim_dt)

        if joint_state:
            state_command_sockets.send_bytes(clproto.encode(joint_state, clproto.MessageType.JOINT_STATE_MESSAGE))
        if cartesian_wrench:
            force_state_sockets.send_bytes(clproto.encode(cartesian_wrench, clproto.MessageType.CARTESIAN_WRENCH_MESSAGE))

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=0.005)

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
