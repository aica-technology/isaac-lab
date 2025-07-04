# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, VelocityPIDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab_assets.custom_actuator_models.actuator_model_parsers import parse_actuator_model

##
# Configuration
##

UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.99,
            "elbow_joint": 2.3,
            "wrist_1_joint": -1.88,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": -1.57,
        },
    ),
    actuators={
        "arm": parse_actuator_model("actuator_models/actuator_model.json")
    },
)

UR5E_CFG_VELOCIY = UR5E_CFG.copy()
UR5E_CFG_VELOCIY.actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=0.0,
            damping=20000.0,
        ),
    }


UR5E_CFG_IK = UR5E_CFG.copy()
UR5E_CFG_IK.actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=50000.0,
            damping=20000.0,
        ),
    }
UR5E_CFG_LOW_LEVEL = UR5E_CFG.copy()
UR5E_CFG_LOW_LEVEL.actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100000,
            damping=50000,
        ),
    }

UR5E_CFG_LOW_LEVEL_PID = UR5E_CFG.copy()
UR5E_CFG_LOW_LEVEL_PID.actuators = {
        "arm": VelocityPIDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=150,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=10000,
            max_integral_error=150,
            delta_time=0.002
        ),
    }