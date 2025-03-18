from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.compliant.compliant_env_cfg import CompliantControlRLCfg
from isaaclab_assets import UR5E_CFG  

@configclass
class UR5eBaseCompliantReachEnvCfg(CompliantControlRLCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.ee_str = "wrist_3_link"

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/"+self.ee_str

        # set rewards body name
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.action_termination_penalty.params["asset_cfg"].body_names = [self.ee_str]

        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_force_pose.body_name = self.ee_str 
        self.commands.ee_force_pose.ranges.pitch = (0, 0)