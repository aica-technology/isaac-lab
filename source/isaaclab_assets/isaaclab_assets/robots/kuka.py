import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, VelocityPIDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

KR210_EFFORT_LIMITS = {
    "kr210_joint_a1": 4100,
    "kr210_joint_a2": 4100,
    "kr210_joint_a3": 4100,
    "kr210_joint_a4": 1250,
    "kr210_joint_a5": 1250,
    "kr210_joint_a6": 700,
}

KUKA_KR210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/usd/robots/kuka/kr210/kr210_cutter.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "kr210_joint_a1": 0.0,
            "kr210_joint_a2": -2,
            "kr210_joint_a3": 2.4,
            "kr210_joint_a4": -1.57,
            "kr210_joint_a5": 1.57,
            "kr210_joint_a6": 0.3
        },
    ),

    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            stiffness=100000,
            damping=50000.0,
        ),
    },
)

KUKA_VEL_KR210_CFG = KUKA_KR210_CFG.copy()
KUKA_VEL_KR210_CFG.actuators["arm"].stiffness = 0
KUKA_VEL_KR210_CFG.actuators["arm"].damping = 50000

KUKA_KR210_LOW_LEVEL_PID_CFG = KUKA_KR210_CFG.copy()

KUKA_KR210_LOW_LEVEL_PID_CFG.actuators = {
    "arm": VelocityPIDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=KR210_EFFORT_LIMITS,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=20000,
            max_integral_error=KR210_EFFORT_LIMITS,
            delta_time=0.002
        ),
    }
