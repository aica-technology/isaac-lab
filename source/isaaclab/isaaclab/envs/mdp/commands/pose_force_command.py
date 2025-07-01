from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, transform_points

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseForceCommandCfg


class UniformPoseForceCommand(CommandTerm):
    """Command generator for generating force-pose commands uniformly.

    The command generator generates forces and poses by sampling positions uniformly within specified
    regions in cartesian space in x, y while z is fixed to be on a surface. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z). For forces, it samples uniformly
    force setpoints within a predefined range.

    The force, position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseForceCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseForceCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.surface: RigidObject = env.scene[cfg.surface_name]
        self.force_sensor: ContactSensor = env.scene[cfg.force_sensor_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz, fx, fy, fz) in root frame
        self.pose_force_command_b = torch.zeros(self.num_envs, 10, device=self.device)
        self.pose_force_command_b[:, 3] = 1.0
        self.pose_force_command_w = torch.zeros_like(self.pose_force_command_b)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseForceCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 10).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z) and 
        then followed by (fx, fy, fz).
        """
        return self.pose_force_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_force_command_w[:, :3], self.pose_force_command_w[:, 3:7] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_force_command_b[:, :3],
            self.pose_force_command_b[:, 3:7],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_force_command_w[:, :3],
            self.pose_force_command_w[:, 3:7],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )

        force_w, _ = torch.max(torch.mean(self.force_sensor.data.force_matrix_w, dim=1), dim=1) # type: ignore
        ee_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]
        force_ee = transform_points(
            force_w.unsqueeze(1), quat=ee_quat_w
        )

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        self.metrics["exhibited_forces"] = torch.norm(force_ee.squeeze() , dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        surface_height = self.surface.data.body_state_w[env_ids, 0, 2] + 0.15

        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_force_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_force_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_force_command_b[env_ids, 2] = surface_height
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_force_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_force_command_b[env_ids, 3:7] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.pose_force_command_b[env_ids, 7] = r.uniform_(*self.cfg.ranges.force_x)
        self.pose_force_command_b[env_ids, 8] = r.uniform_(*self.cfg.ranges.force_y)
        self.pose_force_command_b[env_ids, 9] = r.uniform_(*self.cfg.ranges.force_z)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose

        # TODO: add a scale for the force visualizer
        self.goal_pose_visualizer.visualize(self.pose_force_command_w[:, :3], self.pose_force_command_w[:, 3:7])
        # -- current body pose
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_state_w[:, :3], body_link_state_w[:, 3:7])