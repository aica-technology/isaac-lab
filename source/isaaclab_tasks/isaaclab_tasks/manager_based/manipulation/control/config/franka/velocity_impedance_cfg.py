from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.control.impedance_scene_cfg import ImpedanceControlRLSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets import FRANKA_PANDA_NO_HAND_CFG


@configclass
class FrankaVelocityImpedanceControlSceneCfg(ImpedanceControlRLSceneCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.ee_str = "panda_hand"

        # override scene
        self.scene.robot = FRANKA_PANDA_NO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/panda_link0"
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + self.ee_str

        # override commands
        self.commands.ee_force_pose.body_name = self.ee_str
        self.commands.ee_force_pose.ranges.pitch = (0, 0)

        # override rewards
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [self.ee_str]

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name=self.ee_str,
            controller=DifferentialIKControllerCfg(command_type="velocity", ik_method="dls"),
            scale=0.02,
            clip=(-0.09, 0.09)
        )
        self.scene.contact_sensor.prim_path = "{ENV_REGEX_NS}/Robot/panda_link7"