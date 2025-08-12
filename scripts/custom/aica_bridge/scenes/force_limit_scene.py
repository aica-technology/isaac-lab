import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab_assets import KUKA_KR210_LOW_LEVEL_PID_CFG


ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()  # type: ignore
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
ee_frame_cfg.prim_path = "/Visuals/EEFrame"


@configclass
class ForceLimitScene(InteractiveSceneCfg):
    robot: ArticulationCfg = KUKA_KR210_LOW_LEVEL_PID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.6, 1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, max_depenetration_velocity=0.1),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.9, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/custom_tool",
        update_period=0.0,
        history_length=2,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/world",
        debug_vis=False,
        visualizer_cfg=ee_frame_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ee_frame",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0, 0, 0),
                ),
            ),
        ],
    )
