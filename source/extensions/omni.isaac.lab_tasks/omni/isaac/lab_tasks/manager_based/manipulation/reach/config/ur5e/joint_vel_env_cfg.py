# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.ur5e.base_env_cfg import UR5eBaseReachEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UR5E_CFG_VELOCIY  # isort: skip


##
# Environment configuration
##


@configclass
class UR5EJointVelReachEnvCfg(UR5eBaseReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur5e
        self.scene.robot = UR5E_CFG_VELOCIY.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
        )

        # remove joint velocity and actions from the observation
        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None