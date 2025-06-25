# Reinforcement Learning with Isaac Lab

This document provides a step-by-step guide for training a Reinforcement Learning (RL) policy based on neural networks
in a simulated environment using Isaac Lab and exporting the trained policy in ONNX format.

Isaac Lab is a modular framework designed to simplify robotics workflows, including RL, learning from demonstrations,
and motion planning. Built on NVIDIA Isaac Sim, it leverages PhysX simulation to deliver photo-realistic environments
and high-performance capabilities. With end-to-end GPU acceleration, Isaac Lab enables faster and more efficient
training of RL policies.

# Prerequisites

Before training a new policy, begin by cloning the AICA fork of
[Isaac Lab](https://github.com/aica-technology/isaac-lab).

Next, refer to the
[Isaac Lab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
guide for detailed steps on installation and developer setup. Isaac Lab is built on top of Isaac Sim, so first follow
the Isaac Sim installation procedure (which can be done through pip). Afterwards, install Isaac Lab by adding the
necessary development dependencies and running the installation script in the cloned repository
(`./isaaclab.sh --install`).

Once done, verify the installation of Isaac Lab by running the following command:

```shell
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```


# Key Concepts

In (deep) neural network-based reinforcement learning, an **actor**, often represented by a **policy** neural network,
learns to perform a specific task by interacting with an environment and optimizing its behavior (policy parameters)
based on received rewards.

A **policy** is the function (or mapping) from states in the environment to the actions an **actor** should take. In
deep RL, this function is implemented by a neural network whose parameters are tuned to maximize cumulative rewards.
Therefore, when we refer to a **policy** in this context, we specifically mean this decision-making function, while the
neural network itself is just one way of representing the policy.

This process involves the following steps:

1. **Interaction with the Environment**: The actor observes the state of the environment and takes actions according to
   its current policy.

2. **Reward Feedback**: After executing an action in the environment, the resulting interaction is interpreted in the
   form of a reward. This reward signals how favorable the action was toward achieving the overall task objective.

3. **Policy Optimization**: The actor updates its policy (neural network parameters) using RL algorithms (A2C, PPO, ...)
   to maximize cumulative rewards.

4. **Generalization via Neural Networks**: The use of (deep) neural networks allows the RL actor to handle
   high-dimensional state and action spaces, enabling it to solve complex tasks such as robotic control, game playing,
   and autonomous navigation.

Through iterative interactions, the actor learns a (near-)optimal policy that maximizes the reward function, that is,
the actor learns to achieve the desired task efficiently, guided by the rewards it accumulates.

Isaac Lab enables **actors** to perform actions and learn policies within simulated environments, supporting thousands
of parallel instances. This parallelization significantly accelerates training cycles, making it highly efficient for RL
tasks.

To set up a RL environment in Isaac Lab, get familiarized with the following topics.

## Asset Management

Assets are objects defined within a 3D scene and can belong to one of the following categories: (1) Articulated Objects,
(2) Rigid Objects, or (3) Deformable Objects. These assets are represented in the USD (Universal Scene Description)
format.

Predefined assets configurations are located in the following
[directory](../source/isaaclab_assets/isaaclab_assets/robots) in the Isaac Lab repository.

This directory contains a range of manipulator robots, including the Franka Emika Panda, Universal Robot UR5e and UR10,
Kinova JACO2, JACO2 and Gen3, uFactory xArm 6, and Kuka KR210. To define a new asset, a Python module defining the asset configuration 
object must be created within the [lab_assets directory](../source/isaaclab_assets/isaaclab_assets/robots). This module 
should reference a corresponding USD file. For detailed instructions on importing a new robot not included in the
[lab_assets directory](..../source/isaaclab_assets/isaaclab_assets/robots), refer to
[Importing a New Asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html).

### Example of Asset Configuration

[Here](../source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py) is an example of defining
an articulation configuration to set up an asset in a reinforcement learning environment:

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

- `spawn`: Defines the USD file path for the asset and specifies its physical properties, such as rigid body settings
  (e.g., enabling or disabling gravity) and contact sensor activation.
- `init_state`: Configures the initial state of the robot, including specific joint positions, joint velocities, ... to
  initialize the articulation.
- `actuators`: Sets up the robot's actuator model, specifying parameters such as velocity limits, effort limits,
  stiffness, and damping.

In this example, the actuator model is defined using `ImplicitActuatorCfg`. However, actuator models in Isaac Lab can be
either implicit or explicit. For more information on configuring actuators, refer to the
[Actuators in Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html).

It’s important to note that the configuration files merely wrap around the underlying USD assets. Without those USD
assets, the configurations would not function. Consequently, asset configurations such as the KUKA KR210 and uFactory
xArm6 are included in the public fork but rely on the corresponding USD assets, which are stored on the AICA Isaac
Nucleus and can be provided upon request.

### Isaac Nucleus

Isaac Nucleus is part of NVIDIA’s Omniverse platform and serves as a central repository for assets and utilities used in
Isaac Lab. It provides a comprehensive collection of prebuilt USD assets such as objects, tables, and manipulator robots
that greatly simplify scene construction. Additionally, Isaac Nucleus simplifies collaboration by storing and sharing
all necessary files in one location, making it easier for multiple users to work together on robotics simulations and
environments.

As shown in the example articulation configuration, the relevant USD file is stored in Isaac Nucleus and can be accessed
by importing `ISAACLAB_NUCLEUS_DIR` from `isaaclab.utils.assets` in Isaac Lab.

Beyond the default assets, AICA has curated a list of additional resources not included in the default Isaac Sim folder
on Isaac Nucleus, such as the uFactory xArm 6 and KUKA KR210 robots, which can be made available upon request.

## Simulation Environments

Simulation environments are virtual environments with which a RL actor interacts to learn a policy. In Isaac Lab, there
are two types of environments:

1. [Manager-Based RL Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
2. [Direct RL Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)

Manager-Based RL Environments simplify simulation complexity by using managers that handle tasks such as fetching,
updating, and setting the data required by the RL agent (for example, constructing the actor state from observations).
These managers are `InteractiveScene`, `ActionManager`, `ObservationManager`, `RewardManager`, `CurriculumManager`, and
`EventManager`. In contrast, Direct-Based RL Environments offer greater flexibility by requiring users to define
observations, actions, and rewards directly within the task script.

In what comes next, Manager-Based RL Environments will be explored in more details.

### Manager-Based RL Environment

To create your own Manager-Based RL Environment, follow this
[tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html). Below is a
summary of a
[basic environment configuration class](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py):

#### Defining A Scene

```python
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
```

As shown in the configuration above, `ReachEnvCfg` inherits from `ManagerBasedRLEnvCfg` and sets up the various managers
that define the environment. The `ReachSceneCfg` specify the assets in the environment such as the ground, robot, and
lights as demonstrated in the snippet below:

```python
@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
```

The scene specifies the environment's assets and consequently determines its complete state.

#### Defining the Observations

While the environment's **scene** represents its complete state, the **observations** represent 
the subset of this state visible to the agent. The agent uses these observations to choose its actions.

The `ObservationsCfg` class defines the observations provided to the RL actor as a state vector. This state vector can 
contain various details, such as joint positions, joint velocities, or the end-effector pose in robotic arms.

Observations can include multiple terms organized into distinct **observation groups**. Grouping observations in 
this way allows users to specify separate observation spaces. Each observation term within a single group
must have the same dimensions.

In the following example, an observation group named `policy` is defined. Note that the `policy` observation group is required by the RL wrappers.

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        # Observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_orientation = ObsTerm(func=mdp.ee_rotation_in_robot_root_frame)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
```

Each observation term specifies how a particular piece of data, such as joint positions or end-effector orientation,
is 1) retrieved, optionally corrupted with noise (for dynamic randomization), and 2) appended to the RL agent’s input
vector. The `func` parameter identifies a callback function that supplies these values, while the order of terms in the
configuration dictates their arrangement in the final state input.

The `ObsTerm` class is defined as follows:

```python
class ObsTerm(
    func: (...) -> Tensor,
    params: dict[str, Any | SceneEntityCfg] = dict(),
    modifiers: list[ModifierCfg] | None = None,
    noise: NoiseCfg | None = None,
    clip: tuple[float, float] | None = None,
    scale: tuple[float, ...] | float | None = None,
    history_length: int = 0,
    flatten_history_dim: bool = True
)
```

Here, the `func` parameter is a function that returns a tensor with shape `(N, d)`, where `N` is the number of environments 
and `d` is the observation term dimension. Additional parameters for the function can be passed via the `params` dictionary. 
Optional arguments allow you to add noise, clip the observations, scale the observation, and apply other modifications as needed.

#### Defining the Actions

The `ActionsCfg` class defines how the outputs of the neural network are translated into actions applied in the
simulation. 

The `ActionsCfg` configures the action manager that can include multiple `ActionTerm` instances. 
Each action term controls a specific aspect of the environment. For example, a robotic arm could have 
two action terms, one controlling the arm joints and another controlling the gripper. This modular design allows 
users to define customized control schemes tailored to different parts of the environment.

An action term processes and applies the actions sent to the environment, directly controlling the asset it manages. This occurs in two steps:

- **Processing actions:** Executed once per environment step, this step pre-processes the raw actions received from the RL policy.
- **Applying actions:** Executed once per simulation step, this step applies the processed actions to the asset managed by the term.

These action terms depend on the type of control mechanism used to operate the robot. For instance, the
policy’s outputs may represent:

- `Joint Position Control`: The policy outputs joint position setpoints.
- `Joint Velocity Control`: The policy outputs joint velocity setpoints.
- `Impedance Control`: The policy outputs joint torque setpoints.
- `Cartesian Control`: The policy outputs Cartesian pose or Cartesian twist setpoints, often processed through an
  inverse kinematics solver.

Below are examples of different control configurations:

##### Example: Joint Position Control

In this configuration, the policy outputs joint position setpoints that are applied directly to the robot:

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True
    )
```

##### Example: Joint Velocity Control

Here, the policy outputs joint velocity setpoints for controlling the robot's movement:

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True
    )
```

##### Example: Cartesian Twist Control

In this setup, the policy outputs Cartesian twist commands (i.e. velocities in Cartesian space) that are translated into
joint commands using an inverse kinematics solver:

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        body_name="tool0",  # End-effector frame name
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls"
        ),
        scale=1.0,
    )
```
#### Defining the Commands

The `CommandsCfg` class defines commands issued to the articulated assets throughout an episode. These 
commands can directly influence the simulated environment, for example, opening a gripper or setting desired 
controller setpoints. Commands can also guide the RL agent by being included as observation terms 
within the `ObservationsCfg` and integrated into the agent's state vector.

Isaac Lab provides multiple command configurations that can be sampled and passed to the
actor state, such as `UniformPoseCommandCfg`, `UniformVelocityCommandCfg`, `NullCommandCfg`, `UniformPose2dCommandCfg`,
and `TerrainBasedPose2dCommandCfg`.

A user can extend these configurations by creating a custom command config (derived from `CommandTermCfg`) and
implementing its functionality in a inherited class of `CommandTerm`. For example, to sample trajectories, a user might
introduce a `UniformTrajectoryCommandCfg` configuration along with a `UniformTrajectoryCommand` class, where the
specific trajectory generation would be implemented. For reference, an example of pose command sampling can be found
[here](../source/isaaclab/isaaclab/envs/mdp/commands/pose_command.py).

Here is a reference to a `CommandsCfg` that utilises `UniformPoseCommandCfg` for a robotic arm reach task:

```python
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        make_quat_unique=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(-0.3, 0.3),
            pos_z=(0.25, 0.4),
            roll=(-math.pi, -math.pi),
            pitch=MISSING,  # depends on the end-effector axis
            yaw=(-2 * math.pi, 2 * math.pi),
        ),
    )
```

The command `ee_pose` can be sampled as demonstrated in the accompanying example and referenced in an observation term:

```python
pose_command = ObsTerm(
    func=mdp.generated_commands,
    params={"command_name": "ee_pose"}
)
```
where in this case, the `mdp.generated_commands` helper function simply reads out the commands given by the command manager.

#### Defining the Rewards

The `RewardsCfg` class defines the reward term that the actor receives after executing an action in the environment. The
reward value is a scalar that is formed by combining various reward terms with there appropriate scaling factor. The
actor aims to maximize its cumulative reward and accordingly chooses the best actions that would maximize the collected
rewards. The higher a reward term range the more it has effect on the behavior and performance of the actor.

Here is an example of a `RewardsCfg` for a simple end-effector tracking policy.

```python
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

As shown in the `RewardsCfg` above, three different reward terms are defined:

- `end_effector_position_tracking`  
  This term has the largest penalty weight (-6.0). It penalizes any significant deviation between the commanded
  end-effector position and its actual position. By making this penalty the largest, the policy is more strongly driven
  to reduce position error, ensuring precise position tracking of the end-effector.

- `end_effector_orientation_tracking`  
  With a penalty weight of -4, this term penalizes orientation error. While important, it is weighted slightly less than
  position tracking, reflecting the prioritization of end-effector position over orientation for this specific task.
  However, the orientation penalty is still considerable enough to ensure stable and controlled orientations.

- `joint_vel`  
  This term, weighted by -0.0001, imposes a small penalty on joint velocities. Even though the penalty is relatively
  small, it helps discourage unnecessarily high velocities that could result in unsafe or overly aggressive motion. This
  gentle penalty contributes to smoother trajectory execution.

Reward tuning is essential for obtaining robust policies and the desired behaviors. Each `RewTerm` uses a `func`
parameter that returns a reward value, which is then multiplied by the term's weight before being combined with other
reward terms.

#### MDP module

The `mdp` (Markov Decision Process) module provides convenient helper functions and classes for defining common 
components used in reinforcement learning, including observations, actions, rewards, curricula, and commands. 
Leveraging these helpers simplifies configuration and promotes consistency across environments. The full list of 
available functions and classes can be found [here](source/isaaclab/isaaclab/envs/mdp).

#### Additional Examples

To explore additional `ManagerBasedRLEnv` examples, consider the following:

1. [Reach Environment](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py)  
   A Robotic arm learning to reach a specified end-effector position and orientation.

2. [Lift Environment](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py)  
   A Robotic arm learning to lift an object within the workspace.

3. [Stack Environment](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py)  
   A Robotic arm learning to stack multiple objects.

For more details on complex reward terms, such as triggering penalties when an object falls out of the workspace, and on
curriculum learning refer to the official
[Isaac Lab Manager-Based RL Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
documentation.

## Reinforcement Learning Workflows

Several open-source RL frameworks facilitate policy learning and optimization. Isaac Lab offers wrappers for various RL
libraries, translating environment data into the appropriate formats for each library's functions. The three supported
libraries are:

1. `RSL-RL`: Developed by the Robotic Systems Lab (RSL) at ETH Zürich, RSL-RL is a fast and straightforward
   implementation of RL algorithms designed to run entirely on GPUs. It currently supports Proximal Policy Optimization
   (PPO), with plans to incorporate additional algorithms. More details can be found on
   [RSL-RL Github](https://github.com/leggedrobotics/rsl_rl).

2. `SKRL`: An open-source modular RL library built on top of PyTorch and JAX. SKRL emphasizes modularity, readability,
   and simplicity, supporting various environments, including NVIDIA Isaac Gym and Omniverse Isaac Gym. It enables
   simultaneous training of multiple agents with customizable scopes. More details can be found on
   [SKRL Github](https://github.com/Toni-SM/skrl).

3. `RL Games`: An open-source high-performance RL framework designed for training policies in simulated environments.
   More details can be found on [RL Games Github](https://github.com/Toni-SM/skrl).

For detailed instructions on running the various RL libraries, refer to
[Reinforcement Learning Wrappers](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html).

To prepare a RL environment for use with one of the mentioned wrappers, additional configuration files need to be
specified. For an example, check out the
[reach directory](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach) in
the Isaac Lab repository.

We recommend using `RSL-RL` because it supports automatic policy export into ONNX format, which is required for deployment with AICA.

### Step 1: Create a New Folder

Begin by creating a new folder. Its location depends on the simulation environment and the specific task being
performed. For example, for a Manager-Based RL robotic arm manipulation reach task, an appropriate location for that
folder would be in
[here](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach). The folder name
should typically match the asset that the policy is acting on, such as `ur5e`, `kr10`, and so on.

### Step 2: Define the Required Structure

Inside this folder, define the `agents` folder, the configuration file (i.e `[control_type]_env_cfg.py`), and
`__init__.py` file for the Manager-Based RL environment.

The `agents` folder contains the configuration file required by the RL wrapper to learn a policy. This file includes all
the essential hyperparameters for training, such as `learning_rate`, `batch_size`, `max_epochs`, and others.

#### Examples of Configurations

To understand the structure and parameters, explore these configuration examples for training the UR5E robot for a Reach
Task using different RL libraries and refer to the chosen library documentation for more details:

- Using `RSL-RL`:
  [RSL-RL PPO Configuration](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur5e/agents/rsl_rl_ppo_cfg.py)
- Using `RL-Games`:
  [RL-Games PPO Configuration](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur5e/agents/rl_games_ppo_cfg.yaml)
- Using `SK-RL`:
  [SK-RL PPO Configuration](../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur5e/agents/skrl_ppo_cfg.yaml)

### Step 3: Register a Gym Environment

To finalize the integration of the new environment with the RL wrappers defined earlier, register the environment as
shown below:

```python
gym.register(
    id="ENVIRONMENT_ID",  # A unique identifier for the environment, used as a reference during training and evaluation
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # Entry point for the Manager-based RL environment
    disable_env_checker=True,  # Disable the environment checker for custom setups
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR5EJointPosReachEnvCfg,  # Configuration for the environment, inheriting from ManagerBasedRLEnv
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # Path to the RL Games configuration file
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eReachPPORunnerCfg",  # Path to the RSL-RL configuration file
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",  # Path to the SK-RL configuration file
    }
)
```

This registration code should be included in the `__init__.py` file located in the directory specified above.

## Training and Exporting the Reinforcement Learning Policies

After defining the assets, environments, and robot control, creating the necessary training configuration files, and
registering the new Gym environment, train a new policy by running the appropriate training script for the Reinforcement
Learning Wrappers. 

```shell
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task ENVIRONMENT_ID --headless
# run script for playing with 32 environments
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task ENVIRONMENT_ID --num_envs 32 --checkpoint /PATH/TO/model.pth
```

Once training is complete, you can export your model to ONNX format. In RSL-RL, this is straightforward using
`export_policy_as_onnx`. For other libraries or custom policies, you can use the PyTorch ONNX exporter as described in
the [official documentation](https://pytorch.org/docs/stable/onnx.html). Note that extra attention is required for
recurrent actor-critic models. Below are example snippets showing how to export both recurrent and feed-forward
actor-critic models:

### Recurrent model (Memory Based Models):

```python
obs = torch.zeros(1, self.rnn.input_size)
h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

actions, h_out, c_out = self(obs, h_in, c_in)

torch.onnx.export(
    self,
    (obs, h_in, c_in),
    os.path.join(path, filename),
    export_params=True,
    opset_version=11,
    verbose=self.verbose,
    input_names=["obs", "h_in", "c_in"],
    output_names=["actions", "h_out", "c_out"],
    dynamic_axes={},
)
```

### Feed-forward model (Common Case):

```python
obs = torch.zeros(1, self.actor[0].in_features)

torch.onnx.export(
    self,
    obs,
    os.path.join(path, filename),
    export_params=True,
    opset_version=11,
    verbose=self.verbose,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
)
```

# Next Steps to Execute the Policy on the Robot

With the ONNX policy in place, AICA provides an SDK that enables users to take a simulation-trained policy and deploy it
on real hardware. This
[README](https://github.com/aica-technology/dynamic-components/tree/main/source/advanced_components/rl_policy_components)
provides a step-by-step guide for deploying a learned policy from Isaac Lab Omniverse to various robotic brands and
integrating it into any complex application developed in AICA Studio.
