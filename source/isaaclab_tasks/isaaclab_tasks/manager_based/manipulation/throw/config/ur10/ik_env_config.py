
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.throw.env_cfg import ThrowEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab_assets import UR10_CFG  # isort: skip

##
# Environment configuration
##

@configclass
class UR10ThrowEnvCfg(ThrowEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # end effector frame name
        self.end_effector_frame_name = "ee_link"

        # switch robot to ur10
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # set end-effector frame in scene
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.end_effector_frame_name 

        # arm config
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.end_effector_frame_name,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
        )