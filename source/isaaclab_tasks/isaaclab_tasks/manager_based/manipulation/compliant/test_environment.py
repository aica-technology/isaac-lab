# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the contact sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms, transform_points
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots import UR5E_CFG

ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"

@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    tilted_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(1.25, 1.0, 0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.30), rot=(1.0, 0.0, 0.0, 0.0)
        ),
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

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TiltedWall"],
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]

    # Create controller
    ik_controller_config = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_controller_config, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy() # type: ignore
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["wrist_3_link"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    ee_jacobi_index = robot_entity_cfg.body_ids[0] - 1  # type: ignore

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Initialize variables
    ik_commands = torch.tensor([[0.45, 0.0, 0.35, 0, 1, 0, 0]], device=torch.device("cuda"))
    count = 1  # iteration count

    # reset robot
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # Initialize a flag to control incrementing and decrementing
    incrementing = True
    ik_increment = 0.05  # Value to increment/decrement

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 40 == 0:
            if incrementing:
                ik_commands[0, 2] += ik_increment
                if ik_commands[0, 2] >= 0.6:
                    incrementing = False
            else:
                ik_commands[0, 2] -= ik_increment
                
                if ik_commands[0, 2] <= 0.29:
                    incrementing = True
            count = 0
            joint_pos_des =  robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            ik_controller.reset()
            ik_controller.set_command(ik_commands)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_index, :, robot_entity_cfg.joint_ids]
            ee_pose_in_world = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] # type: ignore
            root_pose_in_world = robot.data.root_state_w[:, 0:7]
            joint_positions = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            # compute frame in root frame
            end_effector_pose_in_base, end_effector_orientation_base = subtract_frame_transforms(
                root_pose_in_world[:, 0:3], root_pose_in_world[:, 3:7], ee_pose_in_world[:, 0:3], ee_pose_in_world[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = ik_controller.compute(end_effector_pose_in_base, end_effector_orientation_base, jacobian, joint_positions)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_in_world = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] # type: ignore
        # update marker positions
        ee_marker.visualize(ee_pose_in_world[:, 0:3], ee_pose_in_world[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        # print information from the sensors
        print("-------------------------------")
        print(scene["contact_forces"])
        force_w, _ = torch.max(torch.mean(scene["contact_forces"].data.force_matrix_w, dim=1), dim=1)
        ee_pos_w = scene["ee_frame"].data.target_pos_w[..., 0, :]
        ee_quat_w = scene["ee_frame"].data.target_quat_w[..., 0, :]

        force_ee = transform_points(
            force_w, quat=ee_quat_w
        )


        print("observation_forces_w", force_w)
        print("observation_force_tcp", force_ee)

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = ContactSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
