from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.control.force_limit_scene_cfg import ForceLimitEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab_assets import KUKA_KR210_LOW_LEVEL_PID_CFG
import math

@configclass
class KR210ForceLimitEnvCfg(ForceLimitEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # increase env spacing for big robots
        self.scene.env_spacing = 5
        self.episode_length_s = 5
        self.ee_str = "ee_frame"

        #adjust the scene
        self.scene.robot = KUKA_KR210_LOW_LEVEL_PID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_sensor.prim_path = "{ENV_REGEX_NS}/Robot/custom_tool"
        self.scene.table.spawn.size = (0.1, 0.1, 0.1) #type: ignore
        self.scene.table.init_state.pos = (2, 0.0, 0.0) # far distance (this will be overriden by the environment)

        # set rewards body name
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.action_termination_penalty.params["asset_cfg"].body_names = [self.ee_str]
        self.rewards.force_direction_reward.params["asset_cfg"].body_names = [self.ee_str]

        # set end-effector frame
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/world"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        self.commands.ee_pose.body_name = self.ee_str
        self.commands.ee_pose.resampling_time_range = (5.0, 5.0)
        self.commands.ee_pose.ranges.pitch = (0, 0)
        
        #self.commands.ee_pose.ranges.roll=(-math.pi - math.pi/6, -math.pi + math.pi/6)
        self.commands.ee_pose.ranges.yaw = (math.pi/2, math.pi/2)

        self.events.reset_robot_joints.params["ee_frame_name"] = self.ee_str
        self.events.reset_robot_joints.params["arm_joint_names"] = ["kr210_joint_a1", "kr210_joint_a2", "kr210_joint_a3", "kr210_joint_a4", "kr210_joint_a5", "kr210_joint_a6"]
        self.events.reset_robot_joints.params["x_range"] = (1.4, 2.0)
        self.events.reset_robot_joints.params["y_range"] = (-0.6, 0.85)
        self.events.reset_robot_joints.params["z_range"] = (0.74, 0.98)

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=0.02,
            clip=[0.06, 0.06, 0.06, 0.25, 0.25, 0.25],
        )
