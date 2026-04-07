from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.kr210.base_env_cfg import KR210BaseReachEnvCfg

from isaaclab_assets import KUKA_VEL_KR210_CFG


@configclass
class KR210ReachEnvCfg(KR210BaseReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = KUKA_VEL_KR210_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=True
        )

        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None
        self.observations.policy.ee_position = None
        self.observations.policy.ee_orientation = None
