# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.
* :obj:`UR5E_CFG`: The UR5e arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration
##

"""Configuration of K210 arm using implicit actuator models."""
KUKA_KR210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/Users/yrh012/KUKA/kr210/kr210.usd", # TODO: replace with NUCLEUS location
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "kr210_joint_a1": -0.6981,
            "kr210_joint_a2": -1.85,
            "kr210_joint_a3": 2.0944,
            "kr210_joint_a4": 1.047,
            "kr210_joint_a5": 0.6458,
            "kr210_joint_a6": -0.925 
        },
    ),

    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            stiffness=0.0,
            damping=50000.0,
        ),
    },
)

"""
Configuration of UR5e arm for inverse kinematic based control.
"""
KUKA_IK_KR210_CFG = KUKA_KR210_CFG.copy()
KUKA_IK_KR210_CFG.actuators["arm"].stiffness = 100000
KUKA_IK_KR210_CFG.actuators["arm"].damping = 20000
