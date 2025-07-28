from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.control.force_limit_scene_cfg import ForceLimitEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab_assets import UR5E_CFG_LOW_LEVEL_PID


@configclass
class UR5eForceLimitEnvCfg(ForceLimitEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UR5E_CFG_LOW_LEVEL_PID.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.ee_str = "wrist_3_link"

        # set rewards body name
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.action_termination_penalty.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.force_direction_reward.params["asset_cfg"].body_names = [self.ee_str]

        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        self.commands.ee_pose.body_name = self.ee_str
        self.commands.ee_pose.ranges.pitch = (0, 0)

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=0.02,
            clip=(-0.06, 0.06),
        )
