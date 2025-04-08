# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG  
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import math
from . import mdp

##
# Scene definition
##

ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"

@configclass
class CompliantControlSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""
    
    robot: ArticulationCfg = MISSING

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path=MISSING,
            debug_vis=False,
            visualizer_cfg=ee_frame_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=MISSING,
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0, 0, 0),
                    ),
                ),
            ],
        )

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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.15), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TiltedWall"],
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_force_pose = mdp.TrackForcePoseCommandCfg(
        asset_name="robot",
        force_sensor_name="contact_forces",
        surface_name="table",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True, # type: ignore
        ranges=mdp.TrackForcePoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5),
            pos_y=(0.0, 0.0),
            roll=(-math.pi, -math.pi),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(0, 0),
            force_x=(0.0, 0.0),
            force_y=(0.0, 0.0),
            force_z=(-100.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        setpoint_error = ObsTerm(func = mdp.setpoint_error, params={"command_name": "ee_force_pose"})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_orientation = ObsTerm(func=mdp.ee_rotation_in_robot_root_frame)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)

        ee_measured_forces = ObsTerm(func=mdp.measured_forces, noise=Unoise(n_min=-0.01, n_max=0.01))
        desired_force_in_contact = ObsTerm(func=mdp.desired_contact_force, params={"command_name": "ee_force_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # task terms
    end_effector_force_tracking = RewTerm(
        func=mdp.force_pose_command_error,
        weight=-24.0,
        params={"command_name": "ee_force_pose"},
    )

    end_effector_force_tracking_fine_grained = RewTerm(
        func=mdp.force_pose_command_error_tanh,
        weight=3.6,
        params={"command_name": "ee_force_pose", "std": 0.1},
    )
    
    end_effector_force_gradient_penalty = RewTerm(
        func=mdp.measured_force_gradient,
        weight=0.0,
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_force_pose"},
    )

    force_limit_penalty = RewTerm(
        func=mdp.force_limit_penalty,
        weight=0.0,
        params={"command_name": "ee_force_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 28800}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 28800}
    )

    end_effector_force_gradient_penalty_level_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effector_force_gradient_penalty", "weight": -0.04, "num_steps": 60000}
    )

    end_effector_force_gradient_penalty_level_2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effector_force_gradient_penalty", "weight": -0.06, "num_steps": 72000}
    )

    """
    force_limit_penalty_level_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "force_limit_penalty", "weight": -0.1, "num_steps": 84000}
    )

    force_limit_penalty_level_2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "force_limit_penalty", "weight": -0.2, "num_steps": 96000}
    )
    """

##
# Environment configuration
##


@configclass
class CompliantControlRLCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: CompliantControlSceneCfg = CompliantControlSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 200.0
