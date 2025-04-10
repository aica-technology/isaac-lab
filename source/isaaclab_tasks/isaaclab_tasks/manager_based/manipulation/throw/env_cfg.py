# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG 
##
# Scene definition
##



@configclass
class ThrowSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[3, 0, -1.05], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
                scale=(2, 2, 2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

    ball =  RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 2.0)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    pass


@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ThrowSceneCfg = ThrowSceneCfg(num_envs=4096, env_spacing=2.5)