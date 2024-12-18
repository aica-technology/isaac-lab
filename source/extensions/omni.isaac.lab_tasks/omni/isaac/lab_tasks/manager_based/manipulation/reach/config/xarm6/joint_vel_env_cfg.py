from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from omni.isaac.lab_assets import UF_XARM6_VELOCITY


@configclass
class XARM6ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UF_XARM6_VELOCITY.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
        )

        self.observations.policy.joint_vel = None
        self.observations.policy.ee_orientation = None
        self.observations.policy.ee_position = None
        self.observations.policy.actions = None

