## Reinforcement Learning with Isaac Lab

This README provides a step-by-step guide for training a Neural Network-based Reinforcement Learning (RL) policy in a simulation environment and exporting the trained policy in ONNX format.

Isaac Lab is a modular framework designed to simplify robotics research workflows, including reinforcement learning, learning from demonstrations, and motion planning. Built on NVIDIA Isaac Sim, it leverages PhysX simulation to deliver photo-realistic environments and high-performance capabilities. With end-to-end GPU acceleration, Isaac Lab enables faster and more efficient training of RL policies.

## Prerequisites

Before training a new policy, begin by cloning the AICA fork of [Isaac Lab](https://github.com/aica-technology/isaac-lab). 

Next, refer to the [Isaac Lab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) guide for comprehensive instructions on installation and developer setup.

## Key Concepts

In the context of Deep Neural Network-based Reinforcement Learning (RL), an **actor**—typically represented by a neural network—learns to perform a specific task by interacting with an environment and optimizing its behavior based on received rewards. This process involves the following steps:

1. **Interaction with the Environment**: The actor observes the state of the environment and takes actions according to its current policy. The policy, often parameterized by a deep neural network, maps observed states to actions.

2. **Reward Feedback**: After executing an action, the environment provides feedback in the form of a reward. This reward signals how favorable the action was toward achieving the overall task objective.

3. **Policy Optimization**: The actor updates its policy using reinforcement learning algorithms to maximize cumulative rewards. 

4. **Generalization via Deep Neural Networks**: The use of deep neural networks allows the RL agent to handle high-dimensional state and action spaces, enabling it to solve complex tasks such as robotic control, game playing, and autonomous navigation.

Through iterative interactions, the actor refines its understanding of the environment and learns to achieve the desired task efficiently, guided by the rewards it accumulates. 

Isaac Lab enables agents to perform actions and execute policies within simulated environments, supporting thousands of parallel instances. This parallelization significantly accelerates training cycles, making it highly efficient for reinforcement learning tasks.

To set up a Reinforcement Learning environment in Isaac Lab, familiarize yourself with the following topics.

### Asset Management

Assets are objects defined within a 3D scene and can belong to one of the following categories: (1) Articulated Objects, (2) Rigid Objects, or (3) Deformable Objects. These assets are represented in the USD (Universal Scene Description) format. 

Predefined assets are located in the directory:  
`source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets`.  

This directory contains a range of manipulator robots, including the Franka Panda, Universal Robot UR5E and UR10, Kinova, uFactory, and Kuka. To define a new asset, an asset configuration file must be created within the predefined directory. This file should reference a corresponding USD file. For detailed instructions on importing a new robot not included in the predefined directory, refer to [Importing a New Asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html).

#### Example of Asset Configuration
Here is an example of defining an articulation configuration to set up an asset in a Reinforcement Learning environment:

```python
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
```

- **`spawn`**: Defines the USD file path for the asset and specifies its physical properties, such as rigid body settings (e.g., enabling or disabling gravity) and contact sensor activation.
- **`init_state`**: Configures the initial state of the robot, including specific joint positions, joint velocities, ... to initialize the articulation.
- **`actuators`**: Sets up the robot's actuator model, specifying parameters such as velocity limits, effort limits, stiffness, and damping.

In this example, the actuator model is defined using `ImplicitActuatorCfg`. However, actuator models in Isaac Lab can be either implicit or explicit. For more information on configuring actuators, refer to the [Actuators in Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html).

#### Isaac Nucleus
Isaac Nucleus is part of NVIDIA’s Omniverse platform and serves as a central repository for assets and utilities used in Isaac Lab. It provides a comprehensive collection of prebuilt USD assets—such as objects, tables, and manipulator robots—that greatly simplify scene construction. Additionally, Isaac Nucleus streamlines collaboration by storing and sharing all necessary files in one location, making it easier for multiple users to work together on robotics simulations and environments.

As shown in the example articulation configuration, the relevant USD file is stored in Isaac Nucleus and can be accessed by importing `ISAACLAB_NUCLEUS_DIR` from `omni.isaac.lab.utils.assets` in Isaac Lab.

Beyond the default assets, AICA has curated a list of additional resources not included in the default Isaac Sim folder on Isaac Nucleus, such as the uFactory xArm 6 and KUKA KR210 robots, which can be made available upon request.

### Simulation Environments

### Reinforcement Learning Backends

### Training Reinforcement Learning Policies

### Evaluating and Exporting Reinforcement Learning Models

## Next Steps to Execute the Policy on the Robot