from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UR5E_CFG  # isort: skip


##
# Environment configuration
##

ee_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_frame_cfg.prim_path = "/Visuals/EEFrame"

@configclass
class UR5eBaseReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override randomization
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.action_termination_penalty.params["asset_cfg"].body_names = ["tool0"]

        self.ee_str = "tool0"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=ee_frame_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + self.ee_str,
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0, 0, 0),
                    ),
                ),
            ],
        )

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.ranges.pitch = (0, 0)