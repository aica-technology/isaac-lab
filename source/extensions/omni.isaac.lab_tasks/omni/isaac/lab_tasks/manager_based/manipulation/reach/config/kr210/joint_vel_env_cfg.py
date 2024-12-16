# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.kr210.base_env_cfg import KR210BaseReachEnvCfg


@configclass
class KR210ReachEnvCfg(KR210BaseReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=True
        )

        # remove last action and joint velocity from observation
        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None
        self.observations.policy.ee_position = None
        self.observations.policy.ee_orientation = None



