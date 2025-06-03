from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.compliant.compliant_env_cfg import CompliantControlRLCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets import UR5E_CFG_VELOCIY  
import math

@configclass
class UR5eCompliantEnvCfg(CompliantControlRLCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.ee_str = "wrist_3_link"

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG_VELOCIY.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]

        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_force_pose.body_name = self.ee_str 
        self.commands.ee_force_pose.ranges.pitch = (0, 0)

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=1
        )