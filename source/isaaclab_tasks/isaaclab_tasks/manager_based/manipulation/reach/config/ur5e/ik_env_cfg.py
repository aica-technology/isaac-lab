from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.ur5e.base_env_cfg import UR5eBaseReachEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_assets import UR5E_CFG_VELOCIY

@configclass
class UR5eReachIKEnvCfg(UR5eBaseReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG_VELOCIY.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=1.0,
        )

        # remove last action, joint velocity, and joint position from observation
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None
