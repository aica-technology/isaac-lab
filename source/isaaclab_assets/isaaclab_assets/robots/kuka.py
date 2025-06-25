import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


KUKA_KR210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/data/kuka/kr210.usd",
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
            stiffness=100000,
            damping=50000.0,
        ),
    },
)


KUKA_VEL_KR210_CFG = KUKA_KR210_CFG.copy()
KUKA_VEL_KR210_CFG.actuators["arm"].stiffness = 0
KUKA_VEL_KR210_CFG.actuators["arm"].damping = 50000
