from dataclasses import MISSING

# simulation scenes
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

# manager imports
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

# markers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

# utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import math

# markov desicion process files
import isaaclab_tasks.manager_based.manipulation.throw.mdp as mdp

# Scene definition
##

# Frame Definitions
ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"

@configclass
class ThrowSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # rigid ball
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.2,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.056),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.76619, 0.16414, 0.28)),
    )

    # bin to throw the ball in
    bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[2.0, 0, -0.90], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
                scale=(4, 4, 4),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=0.0,
                    disable_gravity=False,
                ),
            ),
        )


    # end-effector frame
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

    # robot articulation
    robot: ArticulationCfg = MISSING
##
# MDP settings
##


@configclass
class CommandsCfg:
    ee_pose_velocity = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        make_quat_unique=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.9, 0.95),
            pos_y=(0.1, 0.1),
            pos_z=(0.55, 0.75),
            roll=(-math.pi, -math.pi),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(0, 0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: ActionTerm = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        # robot state
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_orientation = ObsTerm(func=mdp.ee_rotation_in_robot_root_frame)
        ee_linear_velocity = ObsTerm(func=mdp.ee_linear_velocity)

        # bin location
        throwing_location = ObsTerm(func=mdp.bin_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.998, 1.000),
            "velocity_range": (0.0, 0.0),
        },
    )
@configclass
class RewardsCfg:
    # incentivize reaching a throw point
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose_velocity"},
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=3.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose_velocity"},
    )

    end_effector_velocity_tracking = RewTerm(
        func=mdp.velocity_command_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose_velocity"},
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose_velocity"},
    )

    # ensuring reaching goal
    object_goal_target = RewTerm(
        func=mdp.object_near_target,
        weight=100.0
    )

    object_goal_penalty = RewTerm(
        func=mdp.object_goal_distance_penalty,
        weight=-0.01
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # reward on success
    in_bin = RewTerm(func=mdp.is_terminated, weight=1e6)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    object_on_ground = DoneTerm(func=mdp.ball_in_bin, params={"bin_cfg":  SceneEntityCfg("bin"), "ball_cfg": SceneEntityCfg("ball")})

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # ensure smoothness
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 24000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 24000}
    )

    # incentivize velocity on reach point
    end_effector_velocity_tracking_level_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effector_velocity_tracking", "weight": 0.25, "num_steps": 4500}
    )


    # goal targets
    object_goal_target = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_goal_target", "weight": 20.0, "num_steps": 7200}
    )

    object_goal_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_goal_penalty", "weight": -0.1, "num_steps": 7200}
    )
##
# Environment configuration
##


@configclass
class ThrowEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ThrowSceneCfg = ThrowSceneCfg(num_envs=912, env_spacing=6.0)
    
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
        self.episode_length_s = 4.0

        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type=1
        self.sim.physx.max_position_iteration_count=128  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count=1
        self.sim.physx.bounce_threshold_velocity=0.2
        self.sim.physx.friction_offset_threshold=0.1
        self.sim.physx.friction_correlation_distance=0.00625
        self.sim.physx.gpu_max_rigid_contact_count=2**23
        self.sim.physx.gpu_max_rigid_patch_count=2**23
        self.sim.physx.gpu_max_num_partitions=1