from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.xarm6.base_env_cfg import BaseXARM6ReachEnvCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

@configclass
class XARM6ReachIKEnvCfg(BaseXARM6ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
        )
    
        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None
        self.observations.policy.joint_pos = None
    