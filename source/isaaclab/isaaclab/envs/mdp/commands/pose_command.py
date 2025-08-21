# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        if self.cfg.mode not in ["relative", "absolute"]:
            return ValueError("Pose command mode should either relative or absolute.")

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    @staticmethod
    def _exclusion_region_sampling(random_range: torch.Tensor,
                                low: float, high: float,
                                excl_low: float = -0.025, excl_high: float = 0.025) -> torch.Tensor:
        """
        Fill `random_range` (shape: [num_envs, ...]) with uniform samples from
        [low, excl_low] ∪ [excl_high, high], excluding (excl_low, excl_high).

        Args:
            random_range: preallocated tensor to be filled (dtype/device preserved).
            low, high: overall sampling bounds.
            excl_low, excl_high: open interval to exclude.

        Returns:
            The same tensor, filled in-place.
        """
        # Lengths of allowed sub-ranges (clipped to 0 in case exclusion extends beyond bounds)
        left_len = max(0.0, float(excl_low) - float(low))
        right_len = max(0.0, float(high) - float(excl_high))
        total_len = left_len + right_len

        if total_len <= 0:
            raise ValueError("Exclusion region covers or exceeds the sampling range.")

        # Degenerate sides: if one side has zero length, sample from the other directly.
        if left_len == 0.0:
            return random_range.uniform_(excl_high, high)
        if right_len == 0.0:
            return random_range.uniform_(low, excl_low)

        # Vectorized mapping from U(0,1) to the two allowed intervals, with proper proportions.
        p_left = left_len / total_len  # scalar
        u = random_range.uniform_(0.0, 1.0)  # reuse the provided buffer

        mask_left = u < p_left
        mask_right = ~mask_left

        # Left side: map u in [0, p_left) to [low, excl_low]
        if mask_left.any():
            u_left = u[mask_left]
            random_range[mask_left] = low + (u_left / p_left) * left_len

        # Right side: map u in [p_left, 1] to [excl_high, high]
        if mask_right.any():
            u_right = u[mask_right]
            random_range[mask_right] = excl_high + ((u_right - p_left) / (1.0 - p_left)) * right_len

        return random_range

    @staticmethod
    def randomly_offset_position(positions: torch.Tensor,
                                p_noise: float = 0.5,
                                noise_low: float = -0.02,
                                noise_high: float = 0.02,
                                z_constant: float = -1.0):
        """
        For each row i, with prob `p_noise` do: z_i += U(noise_low, noise_high),
        otherwise set: z_i = z_constant.

        positions: [N, 1] (float tensor)
        """
        mask = torch.empty_like(positions).uniform_(0.0, 1.0) < p_noise
        noise = torch.empty_like(positions).uniform_(noise_low, noise_high)
        return torch.where(mask, noise, torch.full_like(positions, z_constant))

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        exclusion_region_low, exclusion_region_high = -0.055, 0.055
        random_range = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = self._exclusion_region_sampling(
            random_range, *self.cfg.ranges.pos_x, exclusion_region_low, exclusion_region_high
        )
        self.pose_command_b[env_ids, 1] = self._exclusion_region_sampling(
            random_range, *self.cfg.ranges.pos_y, exclusion_region_low, exclusion_region_high
        )
        self.pose_command_b[env_ids, 2] = self._exclusion_region_sampling(
            random_range, *self.cfg.ranges.pos_z, exclusion_region_low, exclusion_region_high
        )

        if self.cfg.mode == "relative":
            ee_pos_b = self.robot.data.body_pos_w[:, self.body_idx] - self.robot.data.root_pos_w
            self.pose_command_b[env_ids, 0] += ee_pos_b[env_ids, 0]
            self.pose_command_b[env_ids, 1] += ee_pos_b[env_ids, 1]
            self.pose_command_b[env_ids, 2] += ee_pos_b[env_ids, 2]

        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

        if self.cfg.spawn:
            rigid_body: RigidObject = self._env.scene[self.cfg.spawn.name]
            positions = self.pose_command_w[env_ids, :3].clone()
            # sample random
            random_range = torch.empty(len(env_ids), device=self.device)
            
            positions[:, 0] += random_range.uniform_(-0.02, 0.02)
            positions[:, 1] += random_range.uniform_(-0.02, 0.02)
            positions[:, 2] += self.randomly_offset_position(positions[:, 2])
        
            orientations = torch.zeros((len(env_ids), 4), device=rigid_body.device)
            orientations[:, 0] = 1
            rigid_body.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


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
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
