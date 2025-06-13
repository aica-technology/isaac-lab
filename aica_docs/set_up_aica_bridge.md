# Running Applications in Isaac Sim Using AICA Studio

This document provides a step-by-step guide for running robotic applications built in **AICA Studio** within a simulated
environment using the **Isaac Lab** wrapper and executed on **Isaac Sim**.

**AICA Studio** is a robotics software tool that enables the creation of robotic applications using a visual, data-flow
approach. Applications are constructed from modular blocks namely, **components**, **controllers**, and **hardware interfaces**.

Hardware interfaces serve as bridges between the application and either physical hardware or simulators, enabling
transitions between simulated and real-world deployments.

**Isaac Lab** is a modular framework built on **NVIDIA Isaac Sim**, designed to simplify robotics workflows such as
reinforcement learning (RL), learning from demonstrations, and motion planning. By leveraging the PhysX engine, it
offers photorealistic simulation and GPU-accelerated performance, making it a great option for training and validating
RL policies.

# Motivation

The integration of AICA Studio with Isaac Lab offers a workflow for developing, testing, and deploying robotic
applications. Specifically:

1. **RL Policy Testing** AICA’s RL Policy Component SDK allows developers to deploy Reinforcement Learning
   (RL) models directly onto real hardware through components. These models can be trained in Isaac Lab, and with AICA application
   interacting directly with Isaac Lab, users can validate the trained policies under the same conditions in which they
   were learned.

2. **Reliable Policy Validation** Validating RL policies in Isaac Sim through AICA ensures consistency in simulation
   fidelity. Developers can monitor the behaviors of the trained policies and test the effect of various parameters,
   enabling confident transitions from simulation to real-world deployment.

3. **One-Click Robot Swapping** AICA Studio makes it easy to switch between robot models, whether simulated or
   real—using a simple dropdown menu. This means developers can reuse the exact same application across simulation and
   hardware with no additional development overhead.

4. **Digital Twin COntrol** Beyond RL, AICA empowers users to interact with digital twins of their robots.
   Applications can be authored, tested, and validated entirely in simulation before connecting to actual hardware. This
   greatly accelerates development cycles and enhances safety.

With this integration, users can build complete automation pipelines in Isaac Lab, interact with them using AICA Studio,
validate performance, switch the hardware interface to a real robot, and hit play with no code changes required.

# Prerequisites

Both **Isaac Lab** and **AICA Studio** require Docker to be installed on your host machine. Ensure Docker is properly
installed and running before continuing.

Begin by cloning the AICA fork of [Isaac Lab](https://github.com/aica-technology/isaac-lab).

Once the repository is cloned, build and start the Docker container by running:

```shell
python3 docker/container.py start
```

This command will build the Docker image and start the container in the background.

Next, enter the running container using:

```shell
python3 docker/container.py enter
```

This ensures you are inside a development environment where Isaac Lab and all required dependencies are already
installed.

Once done, verify the installation by running

```shell
python3 scripts/custom/aica_bridge/run_bridge.py --scene basic_scene
```

This will spwan a UR5e robot, a ground plane and lights as can be seen in the image below.

To run **AICA Studio**, you also need to install the **AICA Launcher**. Follow the installation instructions available
in the [official documentation](https://docs.aica.tech/docs/getting-started/installation/installation-and-launch).


