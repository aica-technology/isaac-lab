"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, VelocityPIDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

_FRANKA_FR3_SBTC_INSTANCEABLE_USD = "/workspace/isaaclab/usd/robots/franka/fr3/panda2fr3_instanceable.usd"
_FRANKA_FR3_NO_HAND_INSTANCEABLE_USD = "/workspace/isaaclab/usd/robots/franka/fr3/panda2fr3_no_hand.usd"
_FRANKA_FR3_SBTC_UNSCREWER_INSTANCEABLE_USD = (
    "/workspace/isaaclab/usd/robots/franka/fr3/panda2fr3_unscrewer_instanceable.usd"
)


##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": -0.11,
            "panda_joint2": 0.29,
            "panda_joint3": 0.13,
            "panda_joint4": -1.95,
            "panda_joint5": -0.02,
            "panda_joint6": 2.32,
            "panda_joint7": 0.0,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


FRANKA_PANDA_EFFORT_CFG = FRANKA_PANDA_CFG.copy()  # type: ignore
FRANKA_PANDA_EFFORT_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_EFFORT_CFG.actuators["panda_shoulder"].stiffness = 0.0
FRANKA_PANDA_EFFORT_CFG.actuators["panda_shoulder"].damping = 0.0
FRANKA_PANDA_EFFORT_CFG.actuators["panda_forearm"].stiffness = 0.0
FRANKA_PANDA_EFFORT_CFG.actuators["panda_forearm"].damping = 0.0
"""Configuration of Franka Emika Panda robot with effort control.

This configuration is useful for task-space control using Operational Space Controller.
"""


FRANKA_PANDA_SBTC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_FR3_SBTC_INSTANCEABLE_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # OZHAN: Assuming compensating the gravity
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.05,
            "panda_joint2": -0.04,
            "panda_joint3": 0.22,
            "panda_joint4": -2.12,
            "panda_joint5": 0.02,
            "panda_joint6": 2.11,
            "panda_joint7": 1.07,
            "panda_finger_joint.*": 0.005,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.62,
            stiffness=0.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5,7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=5.26,
            stiffness=0,
            damping=80.0,
        ),
        "panda_forearm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint6"],
            effort_limit_sim=200.0,
            velocity_limit_sim=4.18,
            stiffness=0.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,  # Default 200.0
            velocity_limit_sim=0.2,
            stiffness=2e3,  # Default 2e3
            damping=1e2,  # Default 1e2
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of SBTC Franka Research 3 robot."""


FRANKA_PANDA_EFFORT_SBTC_CFG = FRANKA_PANDA_SBTC_CFG.copy()  # type: ignore
FRANKA_PANDA_EFFORT_SBTC_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_shoulder"].stiffness = 0.0
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_shoulder"].damping = 0.0
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_forearm"].stiffness = 0.0
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_forearm"].damping = 0.0
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_forearm_joint6"].stiffness = 0.0
FRANKA_PANDA_EFFORT_SBTC_CFG.actuators["panda_forearm_joint6"].damping = 0.0
"""Configuration of SBTC Franka Research 3 robot with effort control.

This configuration is useful for task-space control using Operational Space Controller.
"""


FRANKA_PANDA_UNSCREWER_EFFORT_SBTC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_FR3_SBTC_UNSCREWER_INSTANCEABLE_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Factory settings
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(  # Factory settings
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        # WARNING: collision_props needs to be set from the source asset as we cannot modify instanceables
        # collision_props=sim_utils.CollisionPropertiesCfg(  # Factory settings
        #     contact_offset=0.005,
        #     rest_offset=0.0,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.02,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.62,
            stiffness=0.0,
            damping=0.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5,7]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=5.26,
            stiffness=0.0,
            damping=0.0,
        ),
        "panda_forearm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint6"],
            effort_limit_sim=10.0,
            velocity_limit_sim=4.18,
            stiffness=0.0,
            damping=0.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of SBTC Franka Research 3 robot with unscrewer."""

FRANKA_PANDA_NO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_FR3_NO_HAND_INSTANCEABLE_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.05,
            "panda_joint2": -0.04,
            "panda_joint3": 0.22,
            "panda_joint4": -2.12,
            "panda_joint5": 0.02,
            "panda_joint6": 2.11,
            "panda_joint7": 1.07,
        },
    ),
    actuators={
        "panda_shoulder": VelocityPIDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=87,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=0,
            max_integral_error=0,
            delta_time=0.002
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)