from isaaclab.utils import configclass

from ...impedance_scene_cfg import ImpedanceControlRLSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets import UR5E_CFG_LOW_LEVEL_PID


@configclass
class UR5eVelocityImpedanceControlSceneCfg(ImpedanceControlRLSceneCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.ee_str = "wrist_3_link"

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG_LOW_LEVEL_PID.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=1
        )