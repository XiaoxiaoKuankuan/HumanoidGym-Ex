# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import os
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.robots.mrobot.mrobot_base_task import BaseTask
# from humanoid_gym_ex.utils.terrain import Terrain
from humanoid_gym_ex.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from humanoid_gym_ex.utils.helpers import class_to_dict
from humanoid_gym_ex.envs.base.legged_robot_config import LeggedRobotCfg


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.is_amp = False
        self.is_mimic = True
        self._parse_cfg(self.cfg)
        self.num_control = self.cfg.env.num_control
        self.num_notcontrol = self.cfg.env.num_notcontrol
        self.ref_num_notcontrol = self.cfg.env.ref_num_notcontrol
        self.num_policy_actions = len(self.num_control)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.data_length = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions

        self.actions[:, self.num_control] = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print(f"aciton: {self.actions}")
        # idx_ = self.episode_upperbody_left_phase_buf % self.upperbody_pos_length
        # self.actions[:, self.num_notcontrol] = self.upperbody_dof_pos_buffer[idx_][:, self.num_notcontrol] / self.cfg.control.action_scale * 1.2
        # idx_ = self.episode_upperbody_right_phase_buf % self.upperbody_pos_length
        # self.actions[:, 17:] = self.upperbody_dof_pos_buffer[idx_][:, 17:] / self.cfg.control.action_scale
        self.actions[:, self.num_notcontrol] = self._get_noncontrolled_ref_actions()

        # step physics and render each frame
        self.render()
        for i in range(self.cfg.control.decimation):
            if self.cfg.normalization.actions_filter:
                rate_ = (i+1.)/self.cfg.control.decimation
                actions_filter_ = (1. - rate_) * self.last_actions + rate_ * self.actions
            else:
                actions_filter_ = self.actions
            self.torques = self._compute_torques(actions_filter_).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            if self.cfg.domain_rand.sys_delay:
                self.gym.refresh_actor_root_state_tensor(self.sim)

                self.obs_imu_delay_buffer[:,:,1:] = self.obs_imu_delay_buffer[:,:,:-1].clone()
                self.obs_imu_delay_buffer[:,:,0] = self.root_states.clone()

                self.obs_motor_delay_buffer[:,:,1:] = self.obs_motor_delay_buffer[:,:,:-1].clone()
                self.obs_motor_delay_buffer[:,:,0] = torch.cat((self.dof_pos, self.dof_vel), 1).clone()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        if self.is_amp:
            return self.obs_buf, self.privileged_obs_buf, self.disc_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.aux
        elif self.is_mimic:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.aux
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def _get_noncontrolled_ref_actions(self):
        raise NotImplementedError("Mimic task env must implement _get_noncontrolled_ref_actions().")

    def _advance_reference_phase(self):
        raise NotImplementedError("Mimic task env must implement _advance_reference_phase().")

    def _sample_reference_reset_state(self, env_ids):
        raise NotImplementedError("Mimic task env must implement _sample_reference_reset_state().")

    def _get_reset_reference_dof_state(self, env_ids):
        raise NotImplementedError("Mimic task env must implement _get_reset_reference_dof_state().")

    def _apply_reference_root_reset(self, env_ids):
        raise NotImplementedError("Mimic task env must implement _apply_reference_root_reset().")

    def _get_demo_lengths_for_ref_ids(self, ref_ids):
        if hasattr(self, "demo_lengths"):
            return self.demo_lengths[ref_ids]
        return torch.full_like(ref_ids, self.demo_length)

    def _sample_uniform_phase_starts(self, ref_ids):
        """按当前默认策略均匀采样 reset 起点。"""
        demo_lengths = self._get_demo_lengths_for_ref_ids(ref_ids).clamp(min=1)
        max_start = (demo_lengths - 900).clamp(min=1)
        return torch.floor(
            torch.rand(len(ref_ids), device=self.device) * max_start.float()
        ).long()

    def _sample_mixed_phase_starts(self, ref_ids):
        """70% 均匀采样 + 30% hard windows 采样，并只在均匀分支保留 zero-init。"""
        sampled_phase = self._sample_uniform_phase_starts(ref_ids)
        if len(ref_ids) == 0:
            return sampled_phase

        hard_sampling_ratio = float(getattr(self, "hard_sampling_ratio", 0.0))
        hard_windows_by_motion = getattr(self, "hard_phase_windows_by_motion", None)
        hard_motion_has_windows = getattr(self, "hard_motion_has_windows", None)

        use_hard_sampling = (
            hard_sampling_ratio > 0.0
            and hard_windows_by_motion is not None
            and hard_motion_has_windows is not None
        )
        hard_sample_mask = torch.zeros(len(ref_ids), device=self.device, dtype=torch.bool)

        if use_hard_sampling:
            has_windows = hard_motion_has_windows[ref_ids]
            hard_draw = torch.rand(len(ref_ids), device=self.device) < hard_sampling_ratio
            hard_sample_mask = has_windows & hard_draw

            hard_local_indices = hard_sample_mask.nonzero(as_tuple=False).flatten()
            for local_idx in hard_local_indices.tolist():
                ref_id = int(ref_ids[local_idx].item())
                motion_windows = hard_windows_by_motion[ref_id]
                if len(motion_windows) == 0:
                    continue

                # 先按 window 权重抽一个难点片段，再在该片段对应的采样区间内均匀采起点。
                weights = torch.tensor(
                    [window["weight"] for window in motion_windows],
                    device=self.device,
                    dtype=torch.float,
                )
                selected_window_idx = int(torch.multinomial(weights, 1).item())
                selected_window = motion_windows[selected_window_idx]
                sample_start_min = int(selected_window["sample_start_min"])
                sample_start_max = int(selected_window["sample_start_max"])

                if sample_start_max <= sample_start_min:
                    sampled_phase[local_idx] = sample_start_min
                else:
                    sampled_phase[local_idx] = torch.randint(
                        sample_start_min,
                        sample_start_max + 1,
                        (1,),
                        device=self.device,
                    ).long()[0]

        # 保持原来一部分环境从第 0 帧起步的训练习惯，但只作用在均匀采样分支上，
        # 避免 hard 采样刚抽中又被覆盖回 0。
        uniform_mask = ~hard_sample_mask
        if torch.any(uniform_mask):
            zero_init = torch_rand_float(0.0, 1.0, (len(ref_ids), 1), device=self.device)[:, 0] > 0.5
            zero_init = zero_init & uniform_mask
            sampled_phase[zero_init] = 0

        return sampled_phase

    def _resolve_rigid_body_index_by_name(self, body_name):
        if body_name in self.body_name_to_idx:
            return self.body_name_to_idx[body_name]
        matched_indices = [idx for idx, name in enumerate(self.body_names) if body_name in name]
        if len(matched_indices) == 1:
            return matched_indices[0]
        if len(matched_indices) > 1:
            matched_names = [self.body_names[idx] for idx in matched_indices]
            raise ValueError(f"Rigid body name '{body_name}' matches multiple bodies: {matched_names}")
        raise ValueError(f"Rigid body name '{body_name}' not found in asset bodies: {self.body_names}")
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.curriculum_episode_length_buf += 1
        self.episode_phase_buf = self.episode_phase_buf + 1 #torch.where(self.is_static_stand[:, 0] <= 0.1, self.episode_phase_buf + 1, self.episode_phase_buf * 0)
        # self.episode_upperbody_left_phase_buf = self.episode_upperbody_left_phase_buf + 1
        # self.episode_upperbody_right_phase_buf += self.upperbody_right_speed_timestep
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.computer_aux()
        if self.is_amp:
            self.computer_disc_obs()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self._advance_reference_phase()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_torques = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.base_too_low_buf = self.root_states[:, 2] < 0.5
        current_demo_lengths = self._get_demo_lengths_for_ref_ids(self.ref_idx)
        self.ref_end_reset_buf = self.phase_idx >= (current_demo_lengths - 1)
        # root_height = self.root_states_buffer[self.ref_idx, self.phase_idx][:, 2]

        # self.body_error_buf = torch.abs(self.root_states[:, 2] - root_height) > 0.3

        # diff = self.dof_pos.clone()[:, :self.num_control] - self.dof_pos_buffer[self.ref_idx, self.phase_idx][:, :self.num_control]
        # self.body_error_buf |= torch.norm(diff, dim=1) > 3.

        # euler_xy = self.euler_xyz_buffer[self.ref_idx, self.phase_idx][:, 1]
        # self.body_error_buf |= torch.abs(euler_xy - self.base_euler_xyz[:, 1]) > 1.

        # foot_height = self.foot_height_buffer[self.ref_idx, self.phase_idx]
        # feet_z = self.rigid_state[:, self.feet_indices, 2]
        # self.body_error_buf |= (torch.norm(torch.abs(feet_z - foot_height), dim=1) > 0.06) * (self.episode_phase_buf > 1000)

        # self.reset_buf |= self.body_error_buf
        
        grace_mask = self.episode_length_buf > 5
        self.fall_reset_buf = (termination_contact_buf | self.base_too_low_buf) & grace_mask
        self.reset_buf = self.fall_reset_buf | self.time_out_buf | self.ref_end_reset_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset reference state
        self._sample_reference_reset_state(env_ids)
     
        self._resample_commands(env_ids)

        self._reset_dofs(env_ids)

        self._reset_root_states(env_ids)


        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_torques[env_ids] = 0.

        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (len(env_ids), self.num_actions), device=self.device)

        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        episode_lengths = self.curriculum_episode_length_buf[env_ids].float()
        batch_mean_episode_length = torch.mean(episode_lengths)
        fall_ratio = torch.mean(self.fall_reset_buf[env_ids].float())
        ref_end_ratio = torch.mean(self.ref_end_reset_buf[env_ids].float())
        time_out_ratio = torch.mean(self.time_out_buf[env_ids].float())

        adaptive_min_iteration = getattr(self.cfg.domain_rand, "adaptive_min_iteration", 0)
        if self._adaptive_curriculum_current_iteration >= adaptive_min_iteration:
            ema = getattr(self.cfg.domain_rand, "adaptive_metric_ema", 0.9)
            if self._adaptive_curriculum_resets == 0:
                self._adaptive_curriculum_mean_episode_length = batch_mean_episode_length.item()
                self._adaptive_curriculum_fall_ratio = fall_ratio.item()
            else:
                self._adaptive_curriculum_mean_episode_length = (
                    ema * self._adaptive_curriculum_mean_episode_length
                    + (1.0 - ema) * batch_mean_episode_length.item()
                )
                self._adaptive_curriculum_fall_ratio = (
                    ema * self._adaptive_curriculum_fall_ratio
                    + (1.0 - ema) * fall_ratio.item()
                )
            self._adaptive_curriculum_resets += len(env_ids)
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]/(self.episode_length_buf[env_ids] + 1.))
            self.episode_sums[key][env_ids] = 0.
        if hasattr(self, "tracking_score_sums"):
            for key in self.tracking_score_sums.keys():
                self.extras["episode"]['score_' + key] = torch.mean(
                    self.tracking_score_sums[key][env_ids] / (self.episode_length_buf[env_ids] + 1.)
                )
                self.tracking_score_sums[key][env_ids] = 0.
        self.extras["episode"]["mean_episode_length"] = batch_mean_episode_length
        self.extras["episode"]["fall_ratio"] = fall_ratio
        self.extras["episode"]["ref_end_ratio"] = ref_end_ratio
        self.extras["episode"]["time_out_ratio"] = time_out_ratio
        self.extras["episode"]["curriculum_stage"] = torch.tensor(float(max(self._domain_rand_curriculum_stage, 0)), device=self.device)
        self.extras["episode"]["push_ratio"] = torch.tensor(float(self._current_push_ratio), device=self.device)
        self.extras["episode"]["disturbance_ratio"] = torch.tensor(float(self._current_disturbance_ratio), device=self.device)
        self.extras["episode"]["restitution_ratio"] = torch.tensor(float(self._current_restitution_ratio), device=self.device)
        self.extras["episode"]["pd_ratio"] = torch.tensor(float(self._current_pd_ratio), device=self.device)
        self.extras["episode"]["ankle_pd_ratio"] = torch.tensor(float(self._current_ankle_pd_ratio), device=self.device)
        self.extras["episode"]["motor_strength_ratio"] = torch.tensor(float(self._current_motor_strength_ratio), device=self.device)
        self.extras["episode"]["motor_offset_ratio"] = torch.tensor(float(self._current_motor_offset_ratio), device=self.device)
        self.extras["episode"]["ankle_motor_offset_ratio"] = torch.tensor(float(self._current_ankle_motor_offset_ratio), device=self.device)
        self.extras["episode"]["delay_ratio"] = torch.tensor(float(self._current_delay_ratio), device=self.device)
        self.extras["episode"]["imu_bias_ratio"] = torch.tensor(float(self._current_imu_bias_ratio), device=self.device)
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        self.episode_length_buf[env_ids] = 0
        self.curriculum_episode_length_buf[env_ids] = 0

        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.last_root_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.last_root_offset[env_ids] = 0
        self.last_landing_contacts[env_ids] = False
        self.last_landing_contacts_filt[env_ids] = False
        if getattr(self.cfg.domain_rand, "randomize_default_dof_pos_offset", False):
            offset_range = getattr(self.cfg.domain_rand, "default_dof_pos_offset_range", [-0.01, 0.01])
            self.default_dof_pos_offsets[env_ids] = torch_rand_float(
                offset_range[0],
                offset_range[1],
                (len(env_ids), self.num_actions),
                device=self.device,
            )
            ankle_indices = getattr(self.cfg.domain_rand, "default_dof_pos_offset_ankle_indices", [])
            ankle_range = getattr(self.cfg.domain_rand, "default_dof_pos_offset_ankle_range", None)
            if ankle_range is not None:
                for idx in ankle_indices:
                    self.default_dof_pos_offsets[env_ids, idx] = torch_rand_float(
                        ankle_range[0],
                        ankle_range[1],
                        (len(env_ids), 1),
                        device=self.device,
                    ).squeeze(-1)
        else:
            self.default_dof_pos_offsets[env_ids] = 0.0
        
        #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kp = getattr(self.cfg.domain_rand, 'ankle_kp_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kp is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kp_factors[env_ids, idx] = torch_rand_float(ankle_kp[0], ankle_kp[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kd = getattr(self.cfg.domain_rand, 'ankle_kd_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kd is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kd_factors[env_ids, idx] = torch_rand_float(ankle_kd[0], ankle_kd[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_offset_range[0], self.cfg.domain_rand.motor_offset_range[1], (len(env_ids), self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_motor_offset', False):
            ankle_offset = getattr(self.cfg.domain_rand, 'ankle_motor_offset_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_offset is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.motor_offsets[env_ids, idx] = torch_rand_float(ankle_offset[0], ankle_offset[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_euler_xy_offset:
            self.euler_xy_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.euler_xy_offset_range[0], self.cfg.domain_rand.euler_xy_offset_range[1], (len(env_ids), 2), device=self.device)
        if self.cfg.domain_rand.randomize_euler_z_offset:
            self.euler_z_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.euler_z_offset_range[0], self.cfg.domain_rand.euler_z_offset_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.action_delay:
            self.action_delay_buffer[env_ids, :, :] = 0.0
            self.action_delay_timestep[env_ids] = torch.randint(self.cfg.domain_rand.action_delay_range[0],
                                                        self.cfg.domain_rand.action_delay_range[1], (len(env_ids),),device=self.device)

        if self.cfg.domain_rand.sys_delay:
            self.obs_imu_delay_buffer[env_ids, :, :] = 0.0
            self.obs_motor_delay_buffer[env_ids, :, :] = 0.0

            self.obs_imu_delay_timestep[env_ids] = torch.randint(self.cfg.domain_rand.imu_delay_range[0],
                                                        self.cfg.domain_rand.imu_delay_range[1], (len(env_ids),),device=self.device)
            self.obs_motor_delay_timestep[env_ids] = torch.randint(self.cfg.domain_rand.motor_delay_range[0],
                                                        self.cfg.domain_rand.motor_delay_range[1], (len(env_ids),),device=self.device)

        if self.cfg.domain_rand.randomize_upperbody_speed:
            self.upperbody_left_speed_timestep = torch.randint(self.cfg.domain_rand.upperbody_speed_range[0],
                                                        self.cfg.domain_rand.upperbody_speed_range[1], (self.num_envs,),device=self.device)
            self.upperbody_right_speed_timestep = torch.randint(self.cfg.domain_rand.upperbody_speed_range[0],
                                                        self.cfg.domain_rand.upperbody_speed_range[1], (self.num_envs,),device=self.device)

        self.refresh_actor_rigid_shape_props(env_ids)
        self.refresh_actor_dof_shape_props(env_ids)

        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if name in self.tracking_score_sums:
                self.tracking_score_sums[name] += raw_rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_termination(self):
        """只惩罚真正摔倒，不惩罚超时或参考轨迹正常结束。"""
        return self.fall_reset_buf.float()
        

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id, 0]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id, 0]
        if getattr(self.cfg.domain_rand, "randomize_contact_offsets", False):
            if env_id == 0:
                contact_offset_range = getattr(self.cfg.domain_rand, "contact_offset_range", [0.005, 0.02])
                rest_offset_range = getattr(self.cfg.domain_rand, "rest_offset_range", [0.0, 0.002])
                self.contact_offsets = torch_rand_float(
                    contact_offset_range[0], contact_offset_range[1], (self.num_envs, 1), device=self.device
                )
                self.rest_offsets = torch_rand_float(
                    rest_offset_range[0], rest_offset_range[1], (self.num_envs, 1), device=self.device
                )

            for s in range(len(props)):
                if hasattr(props[s], "contact_offset"):
                    props[s].contact_offset = self.contact_offsets[env_id, 0]
                if hasattr(props[s], "rest_offset"):
                    props[s].rest_offset = self.rest_offsets[env_id, 0]
        return props

    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        if getattr(self.cfg.domain_rand, "randomize_contact_offsets", False):
            contact_offset_range = getattr(self.cfg.domain_rand, "contact_offset_range", [0.005, 0.02])
            rest_offset_range = getattr(self.cfg.domain_rand, "rest_offset_range", [0.0, 0.002])
            self.contact_offsets[env_ids] = torch_rand_float(
                contact_offset_range[0], contact_offset_range[1], (len(env_ids), 1), device=self.device
            )
            self.rest_offsets[env_ids] = torch_rand_float(
                rest_offset_range[0], rest_offset_range[1], (len(env_ids), 1), device=self.device
            )

        self._apply_actor_rigid_shape_props(env_ids)

    def _apply_actor_rigid_shape_props(self, env_ids):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])

            for i in range(len(rigid_shape_props)):
                if hasattr(self, "friction_coeffs"):
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if hasattr(self, "restitution_coeffs"):
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
                if hasattr(self, "contact_offsets") and hasattr(rigid_shape_props[i], "contact_offset"):
                    rigid_shape_props[i].contact_offset = self.contact_offsets[env_id, 0]
                if hasattr(self, "rest_offsets") and hasattr(rigid_shape_props[i], "rest_offset"):
                    rigid_shape_props[i].rest_offset = self.rest_offsets[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], rigid_shape_props)

    @staticmethod
    def _scale_zero_center_range(target_range, ratio):
        return [target_range[0] * ratio, target_range[1] * ratio]

    @staticmethod
    def _scale_one_center_range(target_range, ratio):
        return [1.0 + (target_range[0] - 1.0) * ratio, 1.0 + (target_range[1] - 1.0) * ratio]

    @staticmethod
    def _scale_delay_range(target_range, ratio):
        high = max(1, int(np.ceil(target_range[1] * ratio)))
        low = int(np.floor(target_range[0] * ratio))
        low = min(low, high - 1)
        return [max(low, 0), high]

    def refresh_actor_rigid_body_props(self, env_ids):
        com_offset_x = getattr(self.cfg.domain_rand, 'com_offset_x', 0.)
        com_offset_y = getattr(self.cfg.domain_rand, 'com_offset_y', 0.)
        com_offset_z = getattr(self.cfg.domain_rand, 'com_offset_z', 0.)
        payload_body_idx = getattr(self, "payload_body_index", 0)
        com_body_idx = getattr(self, "com_body_index", 0)

        for env_id in env_ids:
            body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])

            if self.cfg.domain_rand.randomize_payload_mass:
                payload = np.random.uniform(*self.cfg.domain_rand.payload_mass_range)
                self.payload[env_id, 0] = payload
            else:
                payload = 0.0
                self.payload[env_id, 0] = 0.0
            body_props[payload_body_idx].mass = self.default_rigid_body_mass[payload_body_idx] + payload

            if self.cfg.domain_rand.randomize_com_displacement:
                com_x = np.random.uniform(*self.cfg.domain_rand.com_x_pos_range)
                com_y = np.random.uniform(*self.cfg.domain_rand.com_y_pos_range)
                com_z = np.random.uniform(*self.cfg.domain_rand.com_z_pos_range)
                self.com_displacement[env_id, 0] = com_x
                self.com_displacement[env_id, 1] = com_y
                self.com_displacement[env_id, 2] = com_z
            else:
                com_x = 0.0
                com_y = 0.0
                com_z = 0.0
                self.com_displacement[env_id, 0] = com_offset_x
                self.com_displacement[env_id, 1] = com_offset_y
                self.com_displacement[env_id, 2] = com_offset_z

            default_com = self.default_rigid_body_com[com_body_idx]
            body_props[com_body_idx].com = gymapi.Vec3(
                default_com[0].item() + com_x + com_offset_x,
                default_com[1].item() + com_y + com_offset_y,
                default_com[2].item() + com_z + com_offset_z,
            )

            for i in range(len(body_props)):
                if i == payload_body_idx:
                    continue
                if self.cfg.domain_rand.randomize_link_mass:
                    scale = np.random.uniform(*self.cfg.domain_rand.link_mass_range)
                    body_props[i].mass = scale * self.default_rigid_body_mass[i].item()
                else:
                    body_props[i].mass = self.default_rigid_body_mass[i].item()

            self.gym.set_actor_rigid_body_properties(
                self.envs[env_id], self.actor_handles[env_id], body_props, recomputeInertia=True
            )

    def update_domain_rand_curriculum(self, iteration, force=False):
        if not getattr(self.cfg.domain_rand, "use_curriculum", False):
            return
        self._adaptive_curriculum_current_iteration = iteration

        push_schedule = list(getattr(self.cfg.domain_rand, "push_ratio_schedule", [1.0]))
        disturbance_schedule = list(getattr(self.cfg.domain_rand, "disturbance_ratio_schedule", [1.0]))
        restitution_schedule = list(getattr(self.cfg.domain_rand, "restitution_ratio_schedule", [1.0]))
        pd_schedule = list(getattr(self.cfg.domain_rand, "pd_ratio_schedule", [1.0]))
        ankle_pd_schedule = list(getattr(self.cfg.domain_rand, "ankle_pd_ratio_schedule", pd_schedule))
        motor_strength_schedule = list(getattr(self.cfg.domain_rand, "motor_strength_ratio_schedule", [1.0]))
        motor_offset_schedule = list(getattr(self.cfg.domain_rand, "motor_offset_ratio_schedule", [1.0]))
        ankle_motor_offset_schedule = list(getattr(self.cfg.domain_rand, "ankle_motor_offset_ratio_schedule", motor_offset_schedule))
        delay_schedule = list(getattr(self.cfg.domain_rand, "delay_ratio_schedule", [1.0]))
        imu_bias_schedule = list(getattr(self.cfg.domain_rand, "imu_bias_ratio_schedule", [1.0]))
        num_stages = max(
            len(push_schedule),
            len(disturbance_schedule),
            len(restitution_schedule),
            len(pd_schedule),
            len(ankle_pd_schedule),
            len(motor_strength_schedule),
            len(motor_offset_schedule),
            len(ankle_motor_offset_schedule),
            len(delay_schedule),
            len(imu_bias_schedule),
        )
        if num_stages == 0:
            return

        curriculum_mode = getattr(self.cfg.domain_rand, "curriculum_mode", "iteration")
        if curriculum_mode == "adaptive":
            stage_idx = 0
            adaptive_min_iteration = getattr(self.cfg.domain_rand, "adaptive_min_iteration", 0)
            stage_cooldown = getattr(self.cfg.domain_rand, "adaptive_stage_cooldown_iterations", 0)
            current_stage = max(self._domain_rand_curriculum_stage, 0)
            stage_idx = current_stage
            if (
                iteration >= adaptive_min_iteration
                and self._adaptive_curriculum_resets >= getattr(self.cfg.domain_rand, "adaptive_min_resets", 0)
                and (iteration - self._adaptive_curriculum_stage_start_iteration) >= stage_cooldown
            ):
                length_thresholds = list(getattr(self.cfg.domain_rand, "adaptive_length_ratio_thresholds", [0.0] * num_stages))
                fall_thresholds = list(getattr(self.cfg.domain_rand, "adaptive_fall_ratio_thresholds", [1.0] * num_stages))
                mean_length_ratio = self._adaptive_curriculum_mean_episode_length / max(float(self.max_episode_length), 1.0)
                mean_fall_ratio = self._adaptive_curriculum_fall_ratio
                candidate_stage = min(current_stage + 1, num_stages - 1)
                length_threshold = length_thresholds[min(candidate_stage, len(length_thresholds) - 1)]
                fall_threshold = fall_thresholds[min(candidate_stage, len(fall_thresholds) - 1)]
                if mean_length_ratio >= length_threshold and mean_fall_ratio <= fall_threshold:
                    stage_idx = candidate_stage
        else:
            stage_iters = list(getattr(self.cfg.domain_rand, "curriculum_stage_iters", [0]))
            if len(stage_iters) == 0:
                return

            stage_idx = 0
            for idx, start_iter in enumerate(stage_iters):
                if iteration >= start_iter:
                    stage_idx = idx
                else:
                    break

        if (not force) and stage_idx == getattr(self, "_domain_rand_curriculum_stage", -1):
            return

        previous_stage = getattr(self, "_domain_rand_curriculum_stage", -1)

        push_ratio = push_schedule[min(stage_idx, len(push_schedule) - 1)]
        disturbance_ratio = disturbance_schedule[min(stage_idx, len(disturbance_schedule) - 1)]
        restitution_ratio = restitution_schedule[min(stage_idx, len(restitution_schedule) - 1)]
        pd_ratio = pd_schedule[min(stage_idx, len(pd_schedule) - 1)]
        ankle_pd_ratio = ankle_pd_schedule[min(stage_idx, len(ankle_pd_schedule) - 1)]
        motor_strength_ratio = motor_strength_schedule[min(stage_idx, len(motor_strength_schedule) - 1)]
        motor_offset_ratio = motor_offset_schedule[min(stage_idx, len(motor_offset_schedule) - 1)]
        ankle_motor_offset_ratio = ankle_motor_offset_schedule[min(stage_idx, len(ankle_motor_offset_schedule) - 1)]
        delay_ratio = delay_schedule[min(stage_idx, len(delay_schedule) - 1)]
        imu_bias_ratio = imu_bias_schedule[min(stage_idx, len(imu_bias_schedule) - 1)]

        self.cfg.domain_rand.push_robots = self._target_push_robots and push_ratio > 0.0
        self.cfg.domain_rand.max_push_vel_xy = self._target_max_push_vel_xy * push_ratio
        self.cfg.domain_rand.max_push_ang_vel = self._target_max_push_ang_vel * push_ratio

        self.cfg.domain_rand.disturbance = self._target_disturbance and disturbance_ratio > 0.0
        self.cfg.domain_rand.disturbance_range = [
            self._target_disturbance_range[0] * disturbance_ratio,
            self._target_disturbance_range[1] * disturbance_ratio,
        ]

        self.cfg.domain_rand.randomize_restitution = (
            self._target_randomize_restitution and restitution_ratio > 0.0
        )
        self.cfg.domain_rand.restitution_range = [
            self._target_restitution_range[0] * restitution_ratio,
            self._target_restitution_range[1] * restitution_ratio,
        ]

        self.cfg.domain_rand.randomize_kp = self._target_randomize_kp and pd_ratio > 0.0
        self.cfg.domain_rand.kp_range = self._scale_one_center_range(self._target_kp_range, pd_ratio)
        if self._target_ankle_kp_range is not None:
            self.cfg.domain_rand.ankle_kp_range = self._scale_one_center_range(
                self._target_ankle_kp_range, ankle_pd_ratio
            )

        self.cfg.domain_rand.randomize_kd = self._target_randomize_kd and pd_ratio > 0.0
        self.cfg.domain_rand.kd_range = self._scale_one_center_range(self._target_kd_range, pd_ratio)
        if self._target_ankle_kd_range is not None:
            self.cfg.domain_rand.ankle_kd_range = self._scale_one_center_range(
                self._target_ankle_kd_range, ankle_pd_ratio
            )
        self.cfg.domain_rand.randomize_ankle_pd = (
            self._target_randomize_ankle_pd and ankle_pd_ratio > 0.0
        )

        self.cfg.domain_rand.randomize_motor_strength = (
            self._target_randomize_motor_strength and motor_strength_ratio > 0.0
        )
        self.cfg.domain_rand.motor_strength_range = self._scale_one_center_range(
            self._target_motor_strength_range, motor_strength_ratio
        )

        self.cfg.domain_rand.randomize_motor_offset = (
            self._target_randomize_motor_offset and motor_offset_ratio > 0.0
        )
        self.cfg.domain_rand.motor_offset_range = self._scale_zero_center_range(
            self._target_motor_offset_range, motor_offset_ratio
        )
        if self._target_ankle_motor_offset_range is not None:
            self.cfg.domain_rand.ankle_motor_offset_range = self._scale_zero_center_range(
                self._target_ankle_motor_offset_range, ankle_motor_offset_ratio
            )
        self.cfg.domain_rand.randomize_ankle_motor_offset = (
            self._target_randomize_ankle_motor_offset and ankle_motor_offset_ratio > 0.0
        )

        self.cfg.domain_rand.randomize_euler_xy_offset = (
            self._target_randomize_euler_xy_offset and imu_bias_ratio > 0.0
        )
        self.cfg.domain_rand.euler_xy_offset_range = self._scale_zero_center_range(
            self._target_euler_xy_offset_range, imu_bias_ratio
        )

        self.cfg.domain_rand.randomize_euler_z_offset = (
            self._target_randomize_euler_z_offset and imu_bias_ratio > 0.0
        )
        self.cfg.domain_rand.euler_z_offset_range = self._scale_zero_center_range(
            self._target_euler_z_offset_range, imu_bias_ratio
        )

        self.cfg.domain_rand.action_delay = self._target_action_delay and delay_ratio > 0.0
        self.cfg.domain_rand.action_delay_range = self._scale_delay_range(
            self._target_action_delay_range, delay_ratio
        )

        self.cfg.domain_rand.sys_delay = self._target_sys_delay and delay_ratio > 0.0
        self.cfg.domain_rand.imu_delay_range = self._scale_delay_range(
            self._target_imu_delay_range, delay_ratio
        )
        self.cfg.domain_rand.motor_delay_range = self._scale_delay_range(
            self._target_motor_delay_range, delay_ratio
        )

        if self.cfg.domain_rand.randomize_restitution:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.restitution_coeffs[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.restitution_range[0],
                self.cfg.domain_rand.restitution_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            self._apply_actor_rigid_shape_props(all_env_ids)

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device
            )
        else:
            self.Kp_factors[all_env_ids] = 1.0
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kp = getattr(self.cfg.domain_rand, 'ankle_kp_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kp is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kp_factors[all_env_ids, idx] = torch_rand_float(ankle_kp[0], ankle_kp[1], (self.num_envs, 1), device=self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device
            )
        else:
            self.Kd_factors[all_env_ids] = 1.0
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kd = getattr(self.cfg.domain_rand, 'ankle_kd_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kd is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kd_factors[all_env_ids, idx] = torch_rand_float(ankle_kd[0], ankle_kd[1], (self.num_envs, 1), device=self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device
            )
        else:
            self.motor_strength_factors[all_env_ids] = 1.0

        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.motor_offset_range[0], self.cfg.domain_rand.motor_offset_range[1], (self.num_envs, self.num_actions), device=self.device
            )
        else:
            self.motor_offsets[all_env_ids] = 0.0
        if getattr(self.cfg.domain_rand, 'randomize_ankle_motor_offset', False):
            ankle_offset = getattr(self.cfg.domain_rand, 'ankle_motor_offset_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_offset is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.motor_offsets[all_env_ids, idx] = torch_rand_float(ankle_offset[0], ankle_offset[1], (self.num_envs, 1), device=self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_euler_xy_offset:
            self.euler_xy_offset[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.euler_xy_offset_range[0], self.cfg.domain_rand.euler_xy_offset_range[1], (self.num_envs, 2), device=self.device
            )
        else:
            self.euler_xy_offset[all_env_ids] = 0.0

        if self.cfg.domain_rand.randomize_euler_z_offset:
            self.euler_z_offset[all_env_ids] = torch_rand_float(
                self.cfg.domain_rand.euler_z_offset_range[0], self.cfg.domain_rand.euler_z_offset_range[1], (self.num_envs, 1), device=self.device
            )
        else:
            self.euler_z_offset[all_env_ids] = 0.0

        if self._target_action_delay:
            self.action_delay_buffer[all_env_ids, :, :] = 0.0
            if self.cfg.domain_rand.action_delay:
                self.action_delay_timestep[all_env_ids] = torch.randint(
                    self.cfg.domain_rand.action_delay_range[0], self.cfg.domain_rand.action_delay_range[1], (self.num_envs,), device=self.device
                )
            else:
                self.action_delay_timestep[all_env_ids] = 0

        if self._target_sys_delay:
            self.obs_imu_delay_buffer[all_env_ids, :, :] = 0.0
            self.obs_motor_delay_buffer[all_env_ids, :, :] = 0.0
            if self.cfg.domain_rand.sys_delay:
                self.obs_imu_delay_timestep[all_env_ids] = torch.randint(
                    self.cfg.domain_rand.imu_delay_range[0], self.cfg.domain_rand.imu_delay_range[1], (self.num_envs,), device=self.device
                )
                self.obs_motor_delay_timestep[all_env_ids] = torch.randint(
                    self.cfg.domain_rand.motor_delay_range[0], self.cfg.domain_rand.motor_delay_range[1], (self.num_envs,), device=self.device
                )
            else:
                self.obs_imu_delay_timestep[all_env_ids] = 0
                self.obs_motor_delay_timestep[all_env_ids] = 0

        self._domain_rand_curriculum_stage = stage_idx
        if curriculum_mode == "adaptive" and stage_idx != previous_stage:
            self._adaptive_curriculum_stage_start_iteration = iteration
            # 每升一级后重新累计该 stage 的统计，避免旧 EMA/旧样本把后续 stage 一路推满。
            self._adaptive_curriculum_mean_episode_length = 0.0
            self._adaptive_curriculum_fall_ratio = 1.0
            self._adaptive_curriculum_resets = 0
        self._current_push_ratio = push_ratio
        self._current_disturbance_ratio = disturbance_ratio
        self._current_restitution_ratio = restitution_ratio
        self._current_payload_ratio = 1.0
        self._current_com_ratio = 1.0
        self._current_link_mass_ratio = 1.0
        self._current_pd_ratio = pd_ratio
        self._current_ankle_pd_ratio = ankle_pd_ratio
        self._current_motor_strength_ratio = motor_strength_ratio
        self._current_motor_offset_ratio = motor_offset_ratio
        self._current_ankle_motor_offset_ratio = ankle_motor_offset_ratio
        self._current_delay_ratio = delay_ratio
        self._current_imu_bias_ratio = imu_bias_ratio

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            # prepare friction randomization
            self.joint_friction_coeffs = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.joint_armature_coeffs = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

            if self.cfg.domain_rand.randomize_joint_friction:
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                for i in range(self.num_actions):
                    self.joint_friction_coeffs[:, i] = torch_rand_float(joint_friction_range[i][0], joint_friction_range[i][1], (self.num_envs, 1), device=self.device)[:, 0]

            if self.cfg.domain_rand.randomize_joint_armature:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                for i in range(self.num_actions):
                    self.joint_armature_coeffs[:, i] = torch_rand_float(joint_armature_range[i][0], joint_armature_range[i][1], (self.num_envs, 1), device=self.device)[:, 0]

            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit

            if not self.cfg.domain_rand.randomize_joint_armature:
                joint_armature_values = getattr(self.cfg.domain_rand, "joint_armature_values", None)
                for i in range(self.num_actions):
                    if joint_armature_values is not None and i < len(joint_armature_values):
                        self.joint_armature_coeffs[:, i] = float(joint_armature_values[i])
                    else:
                        self.joint_armature_coeffs[:, i] = props["armature"][i].item()

        for i in range(len(props)):
            props["friction"][i] = self.joint_friction_coeffs[env_id, i]
            props["armature"][i] = self.joint_armature_coeffs[env_id, i]
        return props

    def refresh_actor_dof_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = self.cfg.domain_rand.joint_friction_range
            for i in range(self.num_actions):
                self.joint_friction_coeffs[env_ids, i] = torch_rand_float(joint_friction_range[i][0], joint_friction_range[i][1], (len(env_ids), 1), device=self.device)[:, 0]

        if self.cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = self.cfg.domain_rand.joint_armature_range
            for i in range(self.num_actions):
                self.joint_armature_coeffs[env_ids, i] = torch_rand_float(joint_armature_range[i][0], joint_armature_range[i][1], (len(env_ids), 1), device=self.device)[:, 0]
        else:
            joint_armature_values = getattr(self.cfg.domain_rand, "joint_armature_values", None)
            if joint_armature_values is not None:
                for i in range(self.num_actions):
                    if i < len(joint_armature_values):
                        self.joint_armature_coeffs[env_ids, i] = float(joint_armature_values[i])

        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id])

            for i in range(len(dof_props)):
                dof_props["friction"][i] = self.joint_friction_coeffs[env_id, i]
                dof_props["armature"][i] = self.joint_armature_coeffs[env_id, i]

            self.gym.set_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id], dof_props)

    def set_actor_dof_shape_props(self, env_ids, joint_friction, joint_armature):
        # for tune joint_friction and joint_armature
        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id])

            for i in range(len(dof_props)):
                dof_props["friction"][i] = joint_friction[env_id]
                dof_props["armature"][i] = joint_armature[env_id]

            self.gym.set_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id], dof_props)

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        payload_body_idx = getattr(self, "payload_body_index", 0)
        com_body_idx = getattr(self, "com_body_index", 0)
        if self.cfg.domain_rand.randomize_payload_mass:
            props[payload_body_idx].mass = self.default_rigid_body_mass[payload_body_idx] + self.payload[env_id, 0]
        else:
            props[payload_body_idx].mass = self.default_rigid_body_mass[payload_body_idx]
            
        # 质心偏移：支持固定偏移(com_offset_*) + 随机偏移(randomize_com_displacement)
        com_offset_x = getattr(self.cfg.domain_rand, 'com_offset_x', 0.)
        com_offset_y = getattr(self.cfg.domain_rand, 'com_offset_y', 0.)
        com_offset_z = getattr(self.cfg.domain_rand, 'com_offset_z', 0.)
        if self.cfg.domain_rand.randomize_com_displacement:
            rng = self.cfg.domain_rand.com_x_pos_range
            com_x_pos = np.random.uniform(rng[0], rng[1])
            self.com_displacement[env_id,0] = com_x_pos
            rng = self.cfg.domain_rand.com_y_pos_range
            com_y_pos = np.random.uniform(rng[0], rng[1])
            self.com_displacement[env_id,1] = com_y_pos
            rng = self.cfg.domain_rand.com_z_pos_range
            com_z_pos = np.random.uniform(rng[0], rng[1])
            self.com_displacement[env_id,2] = com_z_pos
            default_com = self.default_rigid_body_com[com_body_idx]
            props[com_body_idx].com = gymapi.Vec3(
                default_com[0].item() + com_x_pos + com_offset_x,
                default_com[1].item() + com_y_pos + com_offset_y,
                default_com[2].item() + com_z_pos + com_offset_z,
            )
        elif com_offset_x != 0 or com_offset_y != 0 or com_offset_z != 0:
            # 仅固定偏移（不随机）
            self.com_displacement[env_id, 0] = com_offset_x
            self.com_displacement[env_id, 1] = com_offset_y
            self.com_displacement[env_id, 2] = com_offset_z
            default_com = self.default_rigid_body_com[com_body_idx]
            props[com_body_idx].com = gymapi.Vec3(
                default_com[0].item() + com_offset_x,
                default_com[1].item() + com_offset_y,
                default_com[2].item() + com_offset_z,
            )
        else:
            self.com_displacement[env_id, 0] = 0.0
            self.com_displacement[env_id, 1] = 0.0
            self.com_displacement[env_id, 2] = 0.0
            default_com = self.default_rigid_body_com[com_body_idx]
            props[com_body_idx].com = gymapi.Vec3(
                default_com[0].item(),
                default_com[1].item(),
                default_com[2].item(),
            )

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(len(props)):
                if i == payload_body_idx:
                    continue
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]
        else:
            for i in range(len(props)):
                if i == payload_body_idx:
                    continue
                props[i].mass = self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            if getattr(self.cfg.domain_rand, "randomize_push_interval", False):
                if self.common_step_counter >= self.cfg.domain_rand.next_push_step:
                    self._push_robots()
                    self.cfg.domain_rand.next_push_step = (
                        self.common_step_counter + self._sample_push_interval_steps()
                    )
            elif self.common_step_counter % self.cfg.domain_rand.push_interval == 0:
                self._push_robots()
        elif self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        # walk_ = (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1)
        # self.commands[env_ids, :2] *= walk_.unsqueeze(1)
        # self.commands[env_ids, 2] *= walk_

        # not_static_stand_ = torch_rand_float(0., 1., (len(env_ids), 1), device=self.device)
        # not_static_stand_ = not_static_stand_ > 0.2

        # # resample_ = torch.logical_and(resample_, torch.logical_not(walk_))
        # # commands_ = torch_rand_float(0.2, 0.7, (len(env_ids), 1), device=self.device).squeeze(1)

        # self.commands[env_ids, :] *= not_static_stand_
        # self.is_static_stand[env_ids, :] = 1. - not_static_stand_ * 1.
        # self.base_height_idx[env_ids] = torch.logical_not(not_static_stand_)[:, 0] * torch.randint(0, self.demo_length-1, (len(env_ids), ), device=self.device)

        # self.ref_idx[:] = torch.where(self.is_static_stand[:, 0] > 0., 1, 0)
        # self.phase_idx[:] = torch.where(self.is_static_stand[:, 0] > 0., self.base_height_idx, self.episode_phase_buf % self.demo_length)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains
        d_gains = self.d_gains
        torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = torch_rand_float(-0.15, 0.15, (len(env_ids), self.num_dof), device=self.device) + self.default_dof_pos
        self.dof_vel[env_ids] = torch_rand_float(-0.15, 0.15, (len(env_ids), self.num_dof), device=self.device)*0.
        if self.is_amp or self.is_mimic:
            # random_init_coef_ = torch_rand_float(0., self.rand_init_coef, (len(env_ids), 1), device=self.device) * 0.
           

            # self.dof_pos[env_ids][:, self.num_control] = (self.dof_pos_buffer[0, self.episode_phase_buf[env_ids]][:, self.num_control] + self.dof_pos[env_ids][:, self.num_control] * random_init_coef_)
            # self.dof_vel[env_ids][:, self.num_control] = (self.dof_vel_buffer[0, self.episode_phase_buf[env_ids]][:, self.num_control] + self.dof_vel[env_ids][:, self.num_control] * random_init_coef_)

            env_dof_pos = self.dof_pos[env_ids].clone()
            env_dof_vel = self.dof_vel[env_ids].clone()
            ref_pos, ref_vel = self._get_reset_reference_dof_state(env_ids)
            env_dof_pos[:, self.num_control] = ref_pos[:, self.num_control]
            env_dof_vel[:, self.num_control] = ref_vel[:, self.num_control]
            env_dof_pos[:, self.num_notcontrol] = ref_pos[:, self.ref_num_notcontrol]
            env_dof_vel[:, self.num_notcontrol] = ref_vel[:, self.ref_num_notcontrol]
            # 在参考姿态上叠加一层较小的初始关节姿态扰动，
            # 相当于 qpos0 / init qpos 风格的 reset 噪声。
            if getattr(self.cfg.domain_rand, "randomize_init_dof_pos", False):
                init_dof_pos_range = getattr(self.cfg.domain_rand, "init_dof_pos_range", [-0.05, 0.05])
                rand_noise = torch_rand_float(
                    init_dof_pos_range[0],
                    init_dof_pos_range[1],
                    (len(env_ids), self.num_dof),
                    device=self.device,
                )
                env_dof_pos = env_dof_pos + rand_noise
            self.dof_pos[env_ids] = env_dof_pos
            self.dof_vel[env_ids] = env_dof_vel


        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        self.root_states[env_ids, 7:9] = torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)

        
        if self.is_amp or self.is_mimic:
            random_init_coef_ = torch_rand_float(0., self.rand_init_coef, (len(env_ids), 1), device=self.device)
            # self.root_states[env_ids, 2] *= 0
            self._apply_reference_root_reset(env_ids)
            # self.root_states[env_ids, 3:7] = self.root_states_buffer[0, self.episode_phase_buf[env_ids]][:, 3:7]
            
            # self.root_states[env_ids, 7:13] = (self.root_states_buffer[0, self.episode_phase_buf[env_ids]] + self.root_states[env_ids, :]*random_init_coef_)[:, 7:13]

            # self.root_states[env_ids, 2] += 0.02

            # 在 RSI 到参考相位之后，再给 root 初始状态加一层扰动。
            # 这不是观测噪声，而是 reset 状态噪声：
            # - xy 噪声：让机器人不要总从同一个平面落点起步。
            # - yaw 噪声：让机器人不要总从同一个精确朝向起步。
            if getattr(self.cfg.domain_rand, "randomize_root_xy_reset", False):
                xy_range = getattr(self.cfg.domain_rand, "root_xy_reset_range", [0.0, 0.0])
                xy_noise = torch_rand_float(
                    xy_range[0], xy_range[1], (len(env_ids), 2), device=self.device
                )
                self.root_states[env_ids, 0:2] += xy_noise

            if getattr(self.cfg.domain_rand, "randomize_root_yaw_reset", False):
                yaw_range = getattr(self.cfg.domain_rand, "root_yaw_reset_range", [0.0, 0.0])
                yaw_noise = torch_rand_float(
                    yaw_range[0], yaw_range[1], (len(env_ids), 1), device=self.device
                )[:, 0]
                zero_tensor = torch.zeros_like(yaw_noise)
                yaw_noise_quat = quat_from_euler_xyz(zero_tensor, zero_tensor, yaw_noise)
                self.root_states[env_ids, 3:7] = quat_mul(yaw_noise_quat, self.root_states[env_ids, 3:7])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques_raw = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros_like(self.torques)
        self.curriculum_episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_landing_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_landing_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.key_body_diff = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0


        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")

        default_pd_lines = []
        for i in range(self.num_dofs):
            default_pd_lines.append(
                f"  {self.dof_names[i]}: kp={self.p_gains[0, i].item():.4f}, kd={self.d_gains[0, i].item():.4f}"
            )
        print("Default joint PD gains:")
        print("\n".join(default_pd_lines))

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # reset 原因缓冲区需要在首次 reset_idx() 前存在；首次环境构造时尚未跑 check_termination()
        self.fall_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.ref_end_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.truncation_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)

        if self.is_amp:
            self.disc_history = deque(maxlen=self.cfg.env.d_frame_stack)
            for _ in range(self.cfg.env.d_frame_stack):
                self.disc_history.append(torch.zeros(
                    self.num_envs, self.cfg.env.single_num_disc_obs, dtype=torch.float, device=self.device))

        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance_force = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.euler_xy_offset = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.euler_z_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kp = getattr(self.cfg.domain_rand, 'ankle_kp_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kp is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kp_factors[:, idx] = torch_rand_float(ankle_kp[0], ankle_kp[1], (self.num_envs, 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_pd', False):
            ankle_kd = getattr(self.cfg.domain_rand, 'ankle_kd_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_kd is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kd_factors[:, idx] = torch_rand_float(ankle_kd[0], ankle_kd[1], (self.num_envs, 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets = torch_rand_float(self.cfg.domain_rand.motor_offset_range[0], self.cfg.domain_rand.motor_offset_range[1], (self.num_envs, self.num_actions), device=self.device)
        if getattr(self.cfg.domain_rand, "randomize_default_dof_pos_offset", False):
            offset_range = getattr(self.cfg.domain_rand, "default_dof_pos_offset_range", [-0.01, 0.01])
            self.default_dof_pos_offsets = torch_rand_float(
                offset_range[0],
                offset_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
            ankle_indices = getattr(self.cfg.domain_rand, "default_dof_pos_offset_ankle_indices", [])
            ankle_range = getattr(self.cfg.domain_rand, "default_dof_pos_offset_ankle_range", None)
            if ankle_range is not None:
                for idx in ankle_indices:
                    self.default_dof_pos_offsets[:, idx] = torch_rand_float(
                        ankle_range[0],
                        ankle_range[1],
                        (self.num_envs, 1),
                        device=self.device,
                    ).squeeze(-1)
        if getattr(self.cfg.domain_rand, 'randomize_ankle_motor_offset', False):
            ankle_offset = getattr(self.cfg.domain_rand, 'ankle_motor_offset_range', None)
            ankle_idx = getattr(self.cfg.domain_rand, 'ankle_joint_indices', None)
            if ankle_offset is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.motor_offsets[:, idx] = torch_rand_float(ankle_offset[0], ankle_offset[1], (self.num_envs, 1), device=self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_euler_xy_offset:
            self.euler_xy_offset = torch_rand_float(self.cfg.domain_rand.euler_xy_offset_range[0], self.cfg.domain_rand.euler_xy_offset_range[1], (self.num_envs, 2), device=self.device)
        if self.cfg.domain_rand.randomize_euler_z_offset:
            self.euler_z_offset = torch_rand_float(self.cfg.domain_rand.euler_z_offset_range[0], self.cfg.domain_rand.euler_z_offset_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.action_delay:
            self.action_delay_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.action_delay_range[1],device=self.device)
            self.action_delay_timestep = torch.randint(self.cfg.domain_rand.action_delay_range[0],
                                                        self.cfg.domain_rand.action_delay_range[1], (self.num_envs,),device=self.device)
        if self.cfg.domain_rand.sys_delay:
            self.obs_imu_delay_buffer = torch.zeros(self.num_envs, 13, self.cfg.domain_rand.imu_delay_range[1],device=self.device)
            self.obs_motor_delay_buffer = torch.zeros(self.num_envs, self.num_actions * 2, self.cfg.domain_rand.motor_delay_range[1],device=self.device)

            self.obs_imu_delay_timestep = torch.randint(self.cfg.domain_rand.imu_delay_range[0],
                                                        self.cfg.domain_rand.imu_delay_range[1], (self.num_envs,),device=self.device)
            self.obs_motor_delay_timestep = torch.randint(self.cfg.domain_rand.motor_delay_range[0],
                                                        self.cfg.domain_rand.motor_delay_range[1], (self.num_envs,),device=self.device)

        if self.cfg.domain_rand.randomize_upperbody_speed:
            self.upperbody_left_speed_timestep = torch.randint(self.cfg.domain_rand.upperbody_speed_range[0],
                                                        self.cfg.domain_rand.upperbody_speed_range[1], (self.num_envs,),device=self.device)
            self.upperbody_right_speed_timestep = torch.randint(self.cfg.domain_rand.upperbody_speed_range[0],
                                                        self.cfg.domain_rand.upperbody_speed_range[1], (self.num_envs,),device=self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                if self.is_amp:
                    self.reward_scales[key] /= self.reward_scales_total
                else:
                    self.reward_scales[key] *= self.dt
                
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.tracking_score_names = [
            name
            for name in self.reward_names
            if name.startswith("imitation") or name.startswith("imition") or name == "teleop_contact_mask"
        ]
        self.tracking_score_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.tracking_score_names
        }

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = list(body_names)
        self.body_name_to_idx = {name: idx for idx, name in enumerate(self.body_names)}
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        payload_body_name = getattr(self.cfg.domain_rand, "payload_body_name", self.cfg.asset.base_name)
        com_body_name = getattr(self.cfg.domain_rand, "com_body_name", self.cfg.asset.base_name)
        self.payload_body_index = self._resolve_rigid_body_index_by_name(payload_body_name)
        self.com_body_index = self._resolve_rigid_body_index_by_name(com_body_name)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        ankle_names = [s for s in body_names if self.cfg.asset.ankle_name in s]
        hip_names = [s for s in body_names if self.cfg.asset.hip_name in s]
        pelvic_yaw_names = [s for s in body_names if getattr(self.cfg.asset, "pelvic_yaw_name", "") in s]
        head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        base_names = [s for s in body_names if self.cfg.asset.base_name in s]
        waist_names = [s for s in body_names if self.cfg.asset.waist_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_rigid_body_com = torch.zeros(self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        # if self.cfg.domain_rand.randomize_com_displacement:
        #     self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    self.default_rigid_body_com[j, 0] = body_props[j].com.x
                    self.default_rigid_body_com[j, 1] = body_props[j].com.y
                    self.default_rigid_body_com[j, 2] = body_props[j].com.z
            
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        # print(f"Base link mass in training: {body_props[0].mass}")
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.ankle_indices = torch.zeros(len(ankle_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(ankle_names)):
            self.ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], ankle_names[i])

        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        self.pelvic_yaw_indices = torch.zeros(len(pelvic_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(pelvic_yaw_names)):
            self.pelvic_yaw_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], pelvic_yaw_names[i])

        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(head_names)):
            self.head_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], head_names[i])
        
        self.base_indices = torch.zeros(len(base_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_names)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_names[i])

        self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(waist_names)):
            self.waist_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], waist_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.reward_scales_total = 0
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale >= 0:
                self.reward_scales_total += scale

        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        push_interval_s = self.cfg.domain_rand.push_interval_s
        if isinstance(push_interval_s, (list, tuple, np.ndarray)):
            if len(push_interval_s) != 2:
                raise ValueError("domain_rand.push_interval_s range must be [min_s, max_s]")
            min_interval_s, max_interval_s = float(push_interval_s[0]), float(push_interval_s[1])
            if min_interval_s <= 0 or max_interval_s < min_interval_s:
                raise ValueError("domain_rand.push_interval_s must satisfy 0 < min_s <= max_s")
            self.cfg.domain_rand.randomize_push_interval = True
            self.cfg.domain_rand.push_interval_range = [
                int(np.ceil(min_interval_s / self.dt)),
                int(np.ceil(max_interval_s / self.dt)),
            ]
            self.cfg.domain_rand.next_push_step = self._sample_push_interval_steps()
            self.cfg.domain_rand.push_interval = self.cfg.domain_rand.push_interval_range[1]
        else:
            self.cfg.domain_rand.randomize_push_interval = False
            self.cfg.domain_rand.push_interval = int(np.ceil(push_interval_s / self.dt))
        self.cfg.domain_rand.disturbance_interval = np.ceil(self.cfg.domain_rand.disturbance_s / self.dt)
        self._domain_rand_curriculum_stage = -1
        self._target_push_robots = self.cfg.domain_rand.push_robots
        self._target_disturbance = self.cfg.domain_rand.disturbance
        self._target_randomize_restitution = self.cfg.domain_rand.randomize_restitution
        self._target_randomize_payload_mass = self.cfg.domain_rand.randomize_payload_mass
        self._target_randomize_com_displacement = self.cfg.domain_rand.randomize_com_displacement
        self._target_randomize_link_mass = self.cfg.domain_rand.randomize_link_mass
        self._target_randomize_kp = self.cfg.domain_rand.randomize_kp
        self._target_randomize_kd = self.cfg.domain_rand.randomize_kd
        self._target_randomize_ankle_pd = getattr(self.cfg.domain_rand, "randomize_ankle_pd", False)
        self._target_randomize_motor_strength = self.cfg.domain_rand.randomize_motor_strength
        self._target_randomize_motor_offset = self.cfg.domain_rand.randomize_motor_offset
        self._target_randomize_ankle_motor_offset = getattr(self.cfg.domain_rand, "randomize_ankle_motor_offset", False)
        self._target_action_delay = self.cfg.domain_rand.action_delay
        self._target_sys_delay = self.cfg.domain_rand.sys_delay

        self._target_randomize_euler_xy_offset = self.cfg.domain_rand.randomize_euler_xy_offset
        self._target_randomize_euler_z_offset = self.cfg.domain_rand.randomize_euler_z_offset
        self._target_max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        self._target_max_push_ang_vel = self.cfg.domain_rand.max_push_ang_vel
        self._target_disturbance_range = list(self.cfg.domain_rand.disturbance_range)
        self._target_restitution_range = list(self.cfg.domain_rand.restitution_range)
        self._target_payload_mass_range = list(self.cfg.domain_rand.payload_mass_range)
        self._target_com_x_pos_range = list(self.cfg.domain_rand.com_x_pos_range)
        self._target_com_y_pos_range = list(self.cfg.domain_rand.com_y_pos_range)
        self._target_com_z_pos_range = list(self.cfg.domain_rand.com_z_pos_range)
        self._target_link_mass_range = list(self.cfg.domain_rand.link_mass_range)
        self._target_kp_range = list(self.cfg.domain_rand.kp_range)
        self._target_kd_range = list(self.cfg.domain_rand.kd_range)
        self._target_motor_strength_range = list(self.cfg.domain_rand.motor_strength_range)
        self._target_motor_offset_range = list(self.cfg.domain_rand.motor_offset_range)
        self._target_action_delay_range = list(self.cfg.domain_rand.action_delay_range)
        self._target_imu_delay_range = list(self.cfg.domain_rand.imu_delay_range)
        self._target_motor_delay_range = list(self.cfg.domain_rand.motor_delay_range)
        self._target_euler_xy_offset_range = list(self.cfg.domain_rand.euler_xy_offset_range)
        self._target_euler_z_offset_range = list(self.cfg.domain_rand.euler_z_offset_range)
        self._target_ankle_kp_range = getattr(self.cfg.domain_rand, "ankle_kp_range", None)
        self._target_ankle_kd_range = getattr(self.cfg.domain_rand, "ankle_kd_range", None)
        self._target_ankle_motor_offset_range = getattr(self.cfg.domain_rand, "ankle_motor_offset_range", None)
        self._adaptive_curriculum_mean_episode_length = 0.0
        self._adaptive_curriculum_fall_ratio = 1.0
        self._adaptive_curriculum_resets = 0
        self._adaptive_curriculum_current_iteration = 0
        self._adaptive_curriculum_stage_start_iteration = 0
        self._current_push_ratio = 0.0
        self._current_disturbance_ratio = 0.0
        self._current_restitution_ratio = 0.0
        self._current_payload_ratio = 0.0
        self._current_com_ratio = 0.0
        self._current_link_mass_ratio = 0.0
        self._current_pd_ratio = 0.0
        self._current_ankle_pd_ratio = 0.0
        self._current_motor_strength_ratio = 0.0
        self._current_motor_offset_ratio = 0.0
        self._current_ankle_motor_offset_ratio = 0.0
        self._current_delay_ratio = 0.0
        self._current_imu_bias_ratio = 0.0

    def _sample_push_interval_steps(self):
        min_steps, max_steps = self.cfg.domain_rand.push_interval_range
        return int(np.random.randint(min_steps, max_steps + 1))

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
