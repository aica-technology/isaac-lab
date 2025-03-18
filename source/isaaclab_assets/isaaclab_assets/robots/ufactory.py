import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


UF_XARM6_WITH_GRIPPER = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/data/ufactory_with_gripper/ufactory.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": -1.17,
            "joint4": -1.57,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper.*": 0.00,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            velocity_limit=50.0,
            effort_limit=40.0,
            stiffness={
                "joint1": 1000,
                "joint2": 1000,
                "joint3": 1000,
                "joint4": 1000,
                "joint5": 1000,
                "joint6": 1000,
            },
            damping={
                "joint1": 200,
                "joint2": 200,
                "joint3": 200,
                "joint4": 200,
                "joint5": 200,
                "joint6": 200,           
            }
        ),
        "gripper":ImplicitActuatorCfg(
            joint_names_expr=["gripper.*"],
            velocity_limit=0.2,
            effort_limit=200.0,
            stiffness=2e3,
            damping=1e2
        ),
    },
)

UF_XARM6= ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/data/ufactory/ufactory.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": -1.17,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness={
                "joint1": 800,
                "joint2": 800,
                "joint3": 800,
                "joint4": 800,
                "joint5": 800,
                "joint6": 800
            },
            damping={
                "joint1": 200,
                "joint2": 200,
                "joint3": 200,
                "joint4": 200,
                "joint5": 200,
                "joint6": 200           
            }
        ),
    },
)

UF_XARM6_VELOCITY = UF_XARM6.copy()
UF_XARM6_VELOCITY.actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=0.0,
            damping={
                "joint1": 40,
                "joint2": 40,
                "joint3": 40,
                "joint4": 40,
                "joint5": 40,
                "joint6": 40,                
            }
        ),
    }