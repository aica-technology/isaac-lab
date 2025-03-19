from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.compliant.compliant_env_cfg import CompliantControlRLCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets import UR10_CFG  
import math

@configclass
class UR10CompliantEnvCfg(CompliantControlRLCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.ee_str = "ee_link"

        # switch robot to ur5e
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        self.commands.ee_force_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
        )