import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_UNSCREWER_EFFORT_SBTC_CFG  # isort: skip
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()  # type: ignore
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
ee_frame_cfg.prim_path = "/Visuals/EEFrame"

object_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()  # type: ignore
object_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)  # type: ignore
object_frame_cfg.prim_path = "/Visuals/objectFrame"


@configclass
class UnscrewingScene(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot_base: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/robot_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/usd/objects/tables/franka_base/franka_base_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_UNSCREWER_EFFORT_SBTC_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/fr3"
    )

    robot.init_state.pos = (-0.3712, 0.0, 0.89232)
    robot.spawn.activate_contact_sensors = True  # type: ignore
    robot.actuators["panda_shoulder"].effort_limit_sim = 5
    robot.actuators["panda_forearm"].effort_limit_sim = 5
    robot.actuators["panda_forearm_joint6"].effort_limit_sim = 5


    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple([0.128, 0, 0.99332]), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/usd/objects/screw/screw_instanceable.usd",
            # scale=(0.9, 0.9, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/fr3/panda_link0",
        debug_vis=True,
        visualizer_cfg=ee_frame_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/fr3/unscrewer_tcp",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    object_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/fr3/panda_link0",
        debug_vis=True,
        visualizer_cfg=object_frame_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/object",
                name="object",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    contact_sensor_ee: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3/unscrewer",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object", "{ENV_REGEX_NS}/robot_base"],
        history_length=4,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
