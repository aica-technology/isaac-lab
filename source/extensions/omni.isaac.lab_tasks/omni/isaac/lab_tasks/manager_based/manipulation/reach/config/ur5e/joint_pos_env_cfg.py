# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.ur5e.base_env_cfg import UR5eBaseReachEnvCfg


##
# Environment configuration
##

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

@configclass
class UR5eReachEnvCfg_PLAY(UR5EJointPosReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False