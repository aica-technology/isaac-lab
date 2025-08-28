from dataclasses import MISSING
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
import isaaclab_tasks.manager_based.manipulation.control.mdp as mdp
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

##
# Scene definition
##


ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"


@configclass
class ForceLimitSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, max_depenetration_velocity=0.1),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=2,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        make_quat_unique=True,
        mode="relative",
        spawn=SceneEntityCfg("table"),
        position_only =True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.1, 0.1),
            pos_z=(-0.1, 0.1)
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

        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        #ee_orientation = ObsTerm(func=mdp.ee_rotation_in_robot_root_frame)
        ee_measured_forces = ObsTerm(func=mdp.measured_forces_in_world_frame, noise=Unoise(n_min=-1, n_max=1), params={"scale": 0.01})

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_processed_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_force_compliant_scene,
        mode="reset",
        params={
            "x_range": (0.3, 0.5),
            "y_range": (-0.2, 0.2),
            "z_range": (0.3, 0.5),
            "ee_frame_name": MISSING,
            "arm_joint_names": MISSING
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=3.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )

    """
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    """

    action_termination_penalty = RewTerm(
        func=mdp.action_termination,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    velocity_contact = RewTerm(
        func=mdp.velocity_contact,
        weight=-0.01,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
            "end_effector_cfg": SceneEntityCfg("ee_frame"),
        },
    )

    # force limit penalty
    force_limit_penalty = RewTerm(
        func=mdp.force_limit_penalty,
        weight=-1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
            "end_effector_cfg": SceneEntityCfg("ee_frame"),
            "maximum_limit": 100,
        },
    )

    # force in the direction of the position error
    force_direction_reward = RewTerm(
        func=mdp.force_direction_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor"),
            "command_name": "ee_pose",
            "limit": 1,
        },
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
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.04, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )

    action_termination_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_termination_penalty", "weight": -0.02, "num_steps": 4500},
    )

    force_limit_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "force_limit_penalty", "weight": -2, "num_steps": 6000}
    )


##
# Environment configuration
##


@configclass
class ForceLimitEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the force limit environment."""

    # Scene settings
    scene: ForceLimitSceneCfg = ForceLimitSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


        # self.sim.physx.gpu_max_particle_contacts *= 1
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 500.0
        
        self.sim.physx.max_position_iteration_count = 192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_max_num_partitions = 1  # Important for stable simulation.
        #######################################

        # Custom settings
        self.sim.physx.gpu_collision_stack_size = 2**31
        # self.sim.physx.enable_ccd = True  # TODO Check if this is needed
        # self.sim.physx.gpu_found_lost_pairs_capacity *= 1
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity *= 1
        # self.sim.physx.gpu_total_aggregate_pairs_capacity *= 1
        # self.sim.physx.gpu_heap_capacity *= 1
        # self.sim.physx.gpu_temp_buffer_capacity *= 1
        # self.sim.physx.gpu_max_soft_body_contacts *= 1