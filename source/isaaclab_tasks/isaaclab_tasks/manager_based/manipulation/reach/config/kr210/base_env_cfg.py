from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from isaaclab_assets import KUKA_KR210_CFG

@configclass
class KR210BaseReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.ee_str = "kr210_tool0"

        self.scene.robot = KUKA_KR210_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/kr210_base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str
        
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.action_termination_penalty.params["asset_cfg"].body_names = [self.ee_str]

        # override commands
        self.commands.ee_pose.body_name = self.ee_str
        self.commands.ee_pose.ranges.pos_x = (1.0, 1.5)
        self.commands.ee_pose.ranges.pos_y = (-1.0, 1.0)
        self.commands.ee_pose.ranges.pos_z = (1.0, 2.0)
        self.commands.ee_pose.ranges.pitch = (0, 0)
