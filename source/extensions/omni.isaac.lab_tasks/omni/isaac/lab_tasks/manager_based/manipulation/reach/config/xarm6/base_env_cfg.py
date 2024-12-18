import math

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from omni.isaac.lab_assets import UF_XARM6


@configclass
class BaseXARM6ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e
        self.scene.robot = UF_XARM6.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override randomization
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_eef"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_eef"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_eef"]

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link_eef"
        self.commands.ee_pose.ranges.pitch = (-math.pi, -math.pi)

        # remove last action and joint velocity from observation
        self.observations.policy.joint_vel = None
        self.observations.policy.ee_orientation = None
        self.observations.policy.ee_position = None