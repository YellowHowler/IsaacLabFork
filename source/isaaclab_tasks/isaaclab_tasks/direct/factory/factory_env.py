# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, OBS_DIM_CFG_FORCE, STATE_DIM_CFG_FORCE, FactoryEnvCfg

class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task

        print(OBS_DIM_CFG)
        
        # Update number of obs/states
        if self.cfg_task.name != "four_hole_insert":
            cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
            cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        else:
            cfg.observation_space = sum([OBS_DIM_CFG_FORCE[obs] for obs in cfg.obs_order_force])
            cfg.state_space = sum([STATE_DIM_CFG_FORCE[state] for state in cfg.state_order_force])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
    
        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        if self.cfg_task.name == "peg_insert":
            held_base_z_offset = 0.0
        elif self.cfg_task.name == "four_hole_insert":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            held_base_x_offset = gear_base_offset[0]
            held_base_z_offset = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        elif self.cfg_task.name == "push_t":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        else:
            raise NotImplementedError("Task not implemented")

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "peg_insert":
            self.fixed_success_pos_local[:, 2] = 0.0
        elif self.cfg_task.name == "four_hole_insert":
            self.fixed_success_pos_local[:, 2] = self.cfg_task.fixed_asset_cfg.base_height
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            self.fixed_success_pos_local[:, 0] = gear_base_offset[0]
            self.fixed_success_pos_local[:, 2] = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            head_height = self.cfg_task.fixed_asset_cfg.base_height
            shank_length = self.cfg_task.fixed_asset_cfg.height
            thread_pitch = self.cfg_task.fixed_asset_cfg.thread_pitch
            self.fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
        elif self.cfg_task.name == "push_t":
            self.fixed_success_pos_local[:, 2] = self.cfg_task.fixed_asset_cfg.base_height
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.is_misaligned = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.min_xy_dist = torch.tensor(0.001, device=self.device)
        self.min_z_disp = torch.tensor(0.01, device=self.device)
        self.min_rot_error_angle = torch.tensor(0.05, device=self.device)

        self.prev_force = torch.zeros(self.num_envs, device=self.device)
        

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        if self.cfg_task.name != "four_hole_insert":
            obs_dict = {
                "fingertip_pos": self.fingertip_midpoint_pos,
                "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
                "fingertip_quat": self.fingertip_midpoint_quat,
                "ee_linvel": self.ee_linvel_fd,
                "ee_angvel": self.ee_angvel_fd,
                "prev_actions": prev_actions,
            }

            state_dict = {
                "fingertip_pos": self.fingertip_midpoint_pos,
                "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
                "fingertip_quat": self.fingertip_midpoint_quat,
                "ee_linvel": self.fingertip_midpoint_linvel,
                "ee_angvel": self.fingertip_midpoint_angvel,
                "joint_pos": self.joint_pos[:, 0:7],
                "held_pos": self.held_pos,
                "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
                "held_quat": self.held_quat,
                "fixed_pos": self.fixed_pos,
                "fixed_quat": self.fixed_quat,
                "task_prop_gains": self.task_prop_gains,
                "pos_threshold": self.pos_threshold,
                "rot_threshold": self.rot_threshold,
                "prev_actions": prev_actions,
            }
        else:
            obs_dict = {
                "fingertip_pos": self.fingertip_midpoint_pos,
                "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
                "fingertip_quat": self.fingertip_midpoint_quat,
                # CODE I WROTE
                "joint_torque": self.joint_torque,
                "applied_wrench": self.applied_wrench,
                # END OF CODE I WROTE
                "ee_linvel": self.ee_linvel_fd,
                "ee_angvel": self.ee_angvel_fd,
                "prev_actions": prev_actions,
            }

            state_dict = {
                "fingertip_pos": self.fingertip_midpoint_pos,
                "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
                "fingertip_quat": self.fingertip_midpoint_quat,
                # CODE I WROTE
                "joint_torque": self.joint_torque,
                "applied_wrench": self.applied_wrench,
                # END OF CODE I WROTE
                "ee_linvel": self.fingertip_midpoint_linvel,
                "ee_angvel": self.fingertip_midpoint_angvel,
                "joint_pos": self.joint_pos[:, 0:7],
                "held_pos": self.held_pos,
                "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
                "held_quat": self.held_quat,
                "fixed_pos": self.fixed_pos,
                "fixed_quat": self.fixed_quat,
                "task_prop_gains": self.task_prop_gains,
                "pos_threshold": self.pos_threshold,
                "rot_threshold": self.rot_threshold,
                "prev_actions": prev_actions,
            }
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _check_rot_alignment(self, threshold=0.3):
        """Check if the gripper is aligned with the target orientation."""
        # Compute the quaternion difference (relative rotation)
        relative_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(self.target_held_base_quat), self.held_base_quat)
        
        # Convert quaternion to axis-angle representation
        rot_error_aa = axis_angle_from_quat(relative_quat)
        rot_error_angle = torch.norm(rot_error_aa, p=2, dim=-1) % np.pi  # This is the angle of rotation

        # If the rotational error is within the tolerance, the gripper is aligned
        return_tensor = torch.where(rot_error_angle < threshold, torch.full_like(rot_error_angle, 1.0), torch.full_like(rot_error_angle, 0.0))

        return return_tensor

    def _rotate_gripper_to_target(self):
        """Apply rotation action to gripper to achieve target orientation."""
        # Calculate quaternion difference to rotate the gripper to the target orientation
        rotation_diff = torch_utils.quat_mul(self.held_base_quat, self.target_held_base_quat)

        # Convert the quaternion difference to axis-angle (rotation)
        rotation_aa = axis_angle_from_quat(rotation_diff)
        # Apply this rotation to the gripper
        # The rotation is encoded as an axis-angle action (angle and axis of rotation)
        self.actions[:, 3:6] = rotation_aa  

        # Update gripper target quaternion 
        self.ctrl_target_fingertip_midpoint_quat = self.target_held_base_quat

        self.generate_ctrl_signals()  # Apply the action to rotate the gripper

    def _check_pos_xy_alignment(self, threshold1=0.003, threshold2=0.0055):
        """Check if the gripper's XY position is aligned with the target XY position."""
        xy_dist = torch.norm(self.held_base_pos[:, 0:2] - self.target_held_base_pos[:, 0:2], dim=-1)
        return_tensor = torch.where(xy_dist < threshold1, torch.full_like(xy_dist, 1.0), torch.full_like(xy_dist, 0.0))
        return_tensor = torch.where(
            (xy_dist >= threshold1) & (xy_dist < threshold2),
            torch.full_like(xy_dist, 0.5),
            return_tensor
        )
        return return_tensor
    
    def _move_gripper_to_target_xy(self):
        """Move the gripper to the target XY position using velocity control."""
        # Compute the position error in XY
        xy_error = self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2]

        # Update the action based on the smoothed velocity in XY
        self.actions[:, 0:2] += xy_error  # Move the gripper by the velocity

        # Call the function to apply the action and generate the control signals
        self.generate_ctrl_signals()  # Apply the action to rotate the gripper or move to the target

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # #MY CODE
        # if self.cfg_task.name == "four_hole_insert":
        #     xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        #     centered_envs = xy_dist < 0.005
            
        #     force_drop_threshold = 0.6  # TODO: tune this value based on observation
        #     force_magnitude = torch.abs(self.applied_wrench[:, 2])
        
        #     force_diff = self.prev_force - force_magnitude
        #     force_drop_envs = force_diff > force_drop_threshold

        #     curr_successes = self._get_curr_successes(
        #         success_threshold=self.cfg_task.success_threshold, check_rot=False
        #     )
        #     curr_engaged = self._get_curr_successes(
        #         success_threshold=self.cfg_task.engage_threshold, check_rot=False
        #     )

        #     # Mark environments that experienced a sudden drop
        #     trigger_envs = (centered_envs & force_drop_envs & ~curr_successes).nonzero(as_tuple=False).squeeze(-1)

        #     self.prev_force = force_magnitude.clone()

        #     action[curr_engaged & ~curr_successes, 2] = torch.clamp(action[curr_engaged & ~curr_successes, 2], max=-0.5)
        # #MY CODE END

        # Apply smoothing for the action
        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_height=True, check_rot=False, centered_threshold=0.0025):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        #print(f"z disp: {z_disp}, success threshold: {success_threshold}, xy_dist: {xy_dist}")

        is_centered = torch.where(xy_dist < centered_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "four_hole_insert":
            height_threshold = fixed_cfg.height * success_threshold + self.cfg_task.fixed_asset_cfg.base_height
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        elif self.cfg_task.name == "push_t":
            # No height threshold
            pass
        else:
            raise NotImplementedError("Task not implemented")
        
        curr_successes = is_centered
        
        if check_height:
            is_close_or_below = torch.where(
                z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
            )
        
            curr_successes = torch.logical_and(curr_successes, is_close_or_below)

        if check_rot:
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            
            curr_successes = torch.logical_and(curr_successes, is_rotated)
        
        return curr_successes

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_height = self.cfg_task.name != "push_t"
        check_rot = self.cfg_task.name == "nut_thread" or self.cfg_task.name == "push_t"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, 
            check_height=check_height, check_rot=check_rot
        )

        rew_buf = self._update_rew_buf(curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = {}

        # Keypoint rewards.
        def squashing_fn(x, a, b):
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))
        
        def smooth_reward(x, a, b):
            """Exponential falloff reward"""
            return torch.exp(-a * x) / (b + 1.0)
        
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(self.keypoint_dist, a0, b0)
        # a1, b1 = 25, 2
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(self.keypoint_dist, a1, b1)
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # a2, b2 = 300, 0
        rew_dict["kp_fine"] = squashing_fn(self.keypoint_dist, a2, b2)

        # Action penalties.
        # Action cost (optional)
        action_norm = torch.norm(self.actions, dim=-1, p=2)
        action_delta = torch.norm(self.actions - self.prev_actions, dim=-1, p=2)
        rew_dict["action_penalty"] = action_norm
        rew_dict["action_grad_penalty"] = action_delta

        # Current engagement and successes
        check_height = self.cfg_task.name != "push_t"
        curr_engaged = (
            self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, 
                                     check_height=check_height, check_rot=False)
        )
        rew_dict["curr_engaged"] = curr_engaged.clone().float()
        rew_dict["curr_successes"] = curr_successes.clone().float()

        if self.cfg_task.name == "four_hole_insert":
            # XY distance reward (multi-scale)
            xy_dist = torch.norm(self.held_pos[:, :2] - self.fixed_pos[:, :2], dim=-1)
            xy_dist = torch.max(xy_dist, self.min_xy_dist)

            rew_dict["xy_reward"] = (
                smooth_reward(xy_dist, 30, 0) * 1.0 +
                smooth_reward(xy_dist, 80, 0) * 1.5
            )
            rew_dict["xy_reward"] = torch.clamp(rew_dict["xy_reward"], max=2.0)

            # Rotational reward
            relative_quat = torch_utils.quat_mul(
                torch_utils.quat_conjugate(self.held_quat), self.fixed_quat
            )
            rot_error_aa = axis_angle_from_quat(relative_quat)
            rot_error_angle = torch.norm(rot_error_aa, dim=-1) % torch.pi
            rot_error_angle = torch.min(rot_error_angle, torch.pi - rot_error_angle)
            rot_error_angle = torch.max(rot_error_angle, self.min_rot_error_angle)

            rew_dict["rot_reward"] = smooth_reward(rot_error_angle, 15, 0)
            rew_dict["rot_reward"] = torch.clamp(rew_dict["rot_reward"], max=1.0)

            # Z insertion reward (enabled only if XY is good)
            z_disp = self.held_pos[:, 2] - self.fixed_pos[:, 2]
            z_disp = torch.max(z_disp, self.min_z_disp)

            # xy_gate = torch.clamp((0.05 - xy_dist) / 0.05, 0.0, 1.0)
            # rew_dict["z_reward"] = (0.01 / (0.01 + z_disp)) * xy_gate
            z_weight = torch.clamp(1.0 - xy_dist / 0.05, 0.0, 1.0)
            rew_dict["z_reward"] = z_weight * (0.01 / (0.01 + z_disp))
            rot_gate = torch.clamp((0.3 - rot_error_angle) / 0.3, 0.0, 1.0)
            rew_dict["z_reward"] *= rot_gate

            # Search encouragement (if contacted but not yet engaged)
            cur_contacted = self._get_curr_successes(success_threshold=1.2, check_rot=False, centered_threshold=0.02)
            rew_dict["curr_contacted"] = cur_contacted.clone().float()

            rew_dict["search_reward"] = torch.where(
                (~curr_engaged & cur_contacted),
                action_norm / 12,
                torch.zeros_like(action_norm)
            )

            # if(rew_dict['curr_successes'].sum() > 0.0):
            #     print(f"engaged: {rew_dict['curr_engaged'].sum()}/{self.num_envs}, successes: {rew_dict['curr_successes'].sum()}/{self.num_envs}")

            # Final reward sum
            rew_buf = (
                - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
                - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
                + rew_dict["xy_reward"] * 1.0
                + rew_dict["rot_reward"] * 0.5
                + rew_dict["z_reward"] * 1.5
                + rew_dict["curr_contacted"] * 0.5
                + rew_dict["curr_engaged"] * 2.0
                + rew_dict["curr_successes"] * 5.0
                + rew_dict["search_reward"] * 0.5
            )
        else:
            rew_buf = (
                rew_dict["kp_coarse"]
                + rew_dict["kp_baseline"]
                + rew_dict["kp_fine"]
                - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
                - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
                + rew_dict["curr_engaged"]
                + rew_dict["curr_successes"]
            )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf
    
    def _move_robot_to_target_pos(self, env_id, target_pos, target_rot):
        """
        Move the robot to the desired initial position and orientation before training.
        This ensures the robot starts in the correct pose.
        """
        
        self.actions[env_id, 0:3] = target_pos
        self.actions[env_id, 3:6] = target_rot

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _get_target_gear_base_offset(self):
        """Get offset of target gear from the gear base asset."""
        target_gear = self.cfg_task.target_gear
        if target_gear == "gear_large":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.large_gear_base_offset
        elif target_gear == "gear_medium":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
        elif target_gear == "gear_small":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.small_gear_base_offset
        else:
            raise ValueError(f"{target_gear} not valid in this context!")
        return gear_base_offset

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "four_hole_insert":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            gear_base_offset = self._get_target_gear_base_offset()
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = self.held_base_pos_local
        elif self.cfg_task.name == "push_t":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_quat = self.identity_quat.clone()
            return held_asset_relative_pos, held_asset_relative_quat
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = self.identity_quat
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        if self.cfg_task.name == "four_hole_insert":
            gripper_width = self.cfg_task.held_asset_cfg.width / 2 * 1.25
        elif self.cfg_task.name == "push_t":
            gripper_width = self.cfg_task.held_asset_cfg.width / 2 * 1.25
        else: 
            gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat

        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self._get_target_gear_base_offset()[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
            self.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            pos_error, aa_error = self.set_pos_inverse_kinematics(env_ids=bad_envs)
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1
        
        # TODO: find more convenient area to place
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        relative_quat = torch_utils.quat_mul(
            torch_utils.quat_conjugate(self.held_quat), self.fixed_quat
        )
        rot_error_aa = axis_angle_from_quat(relative_quat)
        rot_error_angle = torch.norm(rot_error_aa, p=2, dim=-1) % (np.pi/2) # The magnitude of the rotation error
        rot_error_angle_comp = torch.tensor(np.pi/2) - rot_error_angle
        rot_error_angle = torch.min(rot_error_angle, rot_error_angle_comp)
        rot_error_angle = torch.max(rot_error_angle, torch.tensor(0.05))
        self.prev_rotational_error = rot_error_angle
        self.max_rotation = rot_error_angle
        
        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        if self.cfg_task.name != "push_t":
            # flip gripper z orientation
            flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
                q1=self.fingertip_midpoint_quat,
                t1=self.fingertip_midpoint_pos,
                q2=flip_z_quat,
                t2=torch.zeros_like(self.fingertip_midpoint_pos),
            )

            # get default gripper in asset transform
            held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
            asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
                held_asset_relative_quat, held_asset_relative_pos
            )

            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
            )

            # Add asset in hand randomization
            rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            self.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            if self.cfg_task.name == "gear_mesh":
                self.held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

            held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
            self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=translated_held_asset_quat,
                t1=translated_held_asset_pos,
                q2=self.identity_quat,
                t2=self.held_asset_pos_noise,
            )

            held_state = self._held_asset.data.default_root_state.clone()
            held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            held_state[:, 3:7] = translated_held_asset_quat
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._held_asset.reset()

            #  Close hand
            # Set gains to use for quick resets.
            reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
                (self.num_envs, 1)
            )
            reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
            self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

            self.step_sim_no_action()

            grasp_time = 0.0
            while grasp_time < 0.25:
                self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
                self.ctrl_target_gripper_dof_pos = 0.0
                self.close_gripper_in_place()
                self.step_sim_no_action()
                grasp_time += self.sim.get_physics_dt()
        else:
            # Set held object on ground
            held_state = self._held_asset.data.default_root_state.clone()

            # Use the fixed asset (marking) position as reference
            base_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins

            # Randomize around it
            rand_xy = 2 * (torch.rand((self.num_envs, 2), device=self.device) - 0.5)  # [-1, 1]
            pos_noise_range = torch.tensor([0.05, 0.05], device=self.device)  # 5cm range
            pos_offset = rand_xy * pos_noise_range

            # Set new held positions on the ground
            held_state[:, 0] = base_pos[:, 0] + pos_offset[:, 0]
            held_state[:, 1] = base_pos[:, 1] + pos_offset[:, 1]
            held_state[:, 2] = base_pos[:, 2]  # same Z as T-marking

            # Orientation randomization (yaw only)
            yaw_angles = 2 * np.pi * torch.rand((self.num_envs,), device=self.device)
            held_quat = torch_utils.quat_from_euler_xyz(
                roll=torch.zeros_like(yaw_angles),
                pitch=torch.zeros_like(yaw_angles),
                yaw=yaw_angles,
            )
            held_state[:, 3:7] = held_quat

            held_state[:, 7:] = 0.0  # zero velocity

            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._held_asset.reset()
            
            self.step_sim_no_action()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        # Back out what actions should be for initial state.
        # Relative position to bolt tip.
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))