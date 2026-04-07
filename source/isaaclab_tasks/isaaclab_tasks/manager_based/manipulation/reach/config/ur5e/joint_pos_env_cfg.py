from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.ur5e.base_env_cfg import UR5eBaseReachEnvCfg

@configclass
class UR5EJointPosReachEnvCfg(UR5eBaseReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # remove joint velocity, end-effector position and orientation from observation
        self.observations.policy.joint_vel = None
        self.observations.policy.ee_orientation = None
        self.observations.policy.ee_position = None