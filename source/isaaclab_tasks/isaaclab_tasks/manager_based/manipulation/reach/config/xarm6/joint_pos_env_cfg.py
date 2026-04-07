from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.xarm6.base_env_cfg import BaseXARM6ReachEnvCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp


@configclass
class XARM6ReachEnvCfg(BaseXARM6ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        self.observations.policy.joint_vel = None
        self.observations.policy.ee_orientation = None
        self.observations.policy.ee_position = None

        