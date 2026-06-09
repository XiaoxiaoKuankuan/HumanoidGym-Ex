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


from humanoid_gym_ex.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import numpy as np
import torch
from humanoid_gym_ex.envs.robots.mrobot.mrobot_legged_robot import LeggedRobot

from humanoid_gym_ex.utils.terrain import HumanoidTerrain
from humanoid_gym_ex.utils.math import wrap_to_pi
# from collections import deque
from humanoid_gym_ex.utils import torch_utils



class MrobotMimicCommonEnv(LeggedRobot):
    """Shared IsaacGym MRobot mimic base.

    BPM/music and specified-trajectory Dance are sibling subclasses.  This base
    owns common robot tensors, action/torque helpers, observation assembly,
    shared tracking-body helpers and shared reward implementations.  Reference
    source details are supplied through task hooks.
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self._pre_legged_robot_init(cfg, sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.num_aux = self.cfg.env.num_aux
        self.rand_init_coef = self.cfg.env.rand_init_coef
        self.num_disc_obs = self.cfg.env.num_disc_obs
        dof_err_w = getattr(self.cfg.rewards, "dof_err_w", [1.0] * len(self.num_control))
        if len(dof_err_w) != len(self.num_control):
            raise ValueError(
                f"cfg.rewards.dof_err_w length must be {len(self.num_control)}, got {len(dof_err_w)}"
            )
        self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)
        self._init_task_reference()

        self.last_root_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.all_tracking_indices = self._resolve_tracking_body_indices(
            getattr(self.cfg.asset, "tracking_body_names", None)
        )
        self._init_tracking_reference_specs()
        
        self.is_static_stand = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float)
        self.base_height_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.ref_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.phase_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.last_root_offset = torch.zeros(
            self.num_envs, 7, device=self.device, dtype=torch.float)
        self.initial_base_yaw = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        
        self._ankle_obs_joint_indices = torch.tensor(
            getattr(self.cfg.domain_rand, "ankle_obs_joint_indices", [4, 5, 10, 11]),
            device=self.device,
            dtype=torch.long,
        )
        n_ankle_obs = len(self._ankle_obs_joint_indices)
        self.ankle_obs_pos_bias = torch.zeros(
            self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float
        )
        self.ankle_obs_vel_bias = torch.zeros(
            self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float
        )
        self._init_ankle_dq_randomization_buffers(n_ankle_obs)
        self._init_task_buffers()
        self._post_task_reference_init()

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _pre_legged_robot_init(self, cfg, sim_params):
        """Hook for task-specific cfg/sim edits before LeggedRobot creates tensors."""

    def _init_task_reference(self):
        """Load task-specific reference source and initialize reference buffers."""
        raise NotImplementedError

    def _init_task_buffers(self):
        """Hook for task-specific runtime buffers allocated after common buffers."""

    def _post_task_reference_init(self):
        """Hook for task-specific diagnostics after common body mappings exist."""

    def compute_ref_state(self):
        raise NotImplementedError

    def _get_actor_reference_extra_obs(self):
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _get_privileged_reference_phase_obs(self, norm_phase):
        return torch.zeros_like(norm_phase)

    def _get_reference_norm_phase(self):
        return torch.zeros(self.num_envs, 1, device=self.device)

    def _advance_reference_phase(self):
        """Advance task-specific reference phase by one policy step."""

    def _resample_reference_commands(self, env_ids):
        """Resample task-specific reference commands/states at reset."""

    def _resample_ankle_obs_bias(self, env_ids):
        """每次 reset 为脚踝观测采样零偏，只用于 actor 输入观测。"""
        if len(env_ids) == 0:
            return

        n = len(self._ankle_obs_joint_indices)

        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_pos_bias", False):
            low, high = getattr(self.cfg.domain_rand, "ankle_obs_pos_bias_range", [-0.02, 0.02])
            self.ankle_obs_pos_bias[env_ids] = torch_rand_float(
                low, high, (len(env_ids), n), device=self.device
            )
        else:
            self.ankle_obs_pos_bias[env_ids] = 0.0

        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_bias", False):
            low, high = getattr(self.cfg.domain_rand, "ankle_obs_vel_bias_range", [-0.3, 0.3])
            self.ankle_obs_vel_bias[env_ids] = torch_rand_float(
                low, high, (len(env_ids), n), device=self.device
            )
        else:
            self.ankle_obs_vel_bias[env_ids] = 0.0

    def _init_ankle_dq_randomization_buffers(self, n_ankle_obs):
        obs_delay_range = getattr(self.cfg.domain_rand, "ankle_obs_vel_delay_range", [0, 0])
        pd_delay_range = getattr(self.cfg.domain_rand, "ankle_pd_dq_delay_range", [0, 0])
        max_obs_delay = int(max(obs_delay_range)) if len(obs_delay_range) > 0 else 0
        max_pd_delay = int(max(pd_delay_range)) if len(pd_delay_range) > 0 else 0

        self.ankle_obs_vel_delay_buffer = torch.zeros(
            self.num_envs, n_ankle_obs, max(1, max_obs_delay + 1), device=self.device, dtype=torch.float
        )
        self.ankle_obs_vel_delay_timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ankle_obs_vel_filter_alpha = torch.ones(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_obs_vel_filtered = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_obs_vel_filter_initialized = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

        self.ankle_pd_dq_delay_buffer = torch.zeros(
            self.num_envs, n_ankle_obs, max(1, max_pd_delay + 1), device=self.device, dtype=torch.float
        )
        self.ankle_pd_dq_delay_timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ankle_pd_dq_filter_alpha = torch.ones(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_pd_dq_filtered = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_pd_dq_filter_initialized = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

    def _sample_ankle_filter_alpha(self, env_ids, cutoff_range):
        low, high = float(cutoff_range[0]), float(cutoff_range[1])
        cutoff = torch_rand_float(
            low,
            high,
            (len(env_ids), len(self._ankle_obs_joint_indices)),
            device=self.device,
        ).clamp(min=0.0)
        return (1.0 - torch.exp(-2.0 * torch.pi * cutoff * self.dt)).clamp(0.0, 1.0)

    def _resample_ankle_dq_randomization(self, env_ids):
        if len(env_ids) == 0:
            return

        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_delay", False):
            low, high = getattr(self.cfg.domain_rand, "ankle_obs_vel_delay_range", [0, 0])
            self.ankle_obs_vel_delay_timestep[env_ids] = torch.randint(
                int(low), int(high) + 1, (len(env_ids),), device=self.device
            )
        else:
            self.ankle_obs_vel_delay_timestep[env_ids] = 0
        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_filter", False):
            cutoff_range = getattr(self.cfg.domain_rand, "ankle_obs_vel_filter_cutoff_range", [8.0, 20.0])
            self.ankle_obs_vel_filter_alpha[env_ids] = self._sample_ankle_filter_alpha(env_ids, cutoff_range)
        else:
            self.ankle_obs_vel_filter_alpha[env_ids] = 1.0
        self.ankle_obs_vel_delay_buffer[env_ids] = 0.0
        self.ankle_obs_vel_filtered[env_ids] = 0.0
        self.ankle_obs_vel_filter_initialized[env_ids] = False

        if getattr(self.cfg.domain_rand, "randomize_ankle_pd_dq_delay", False):
            low, high = getattr(self.cfg.domain_rand, "ankle_pd_dq_delay_range", [0, 0])
            self.ankle_pd_dq_delay_timestep[env_ids] = torch.randint(
                int(low), int(high) + 1, (len(env_ids),), device=self.device
            )
        else:
            self.ankle_pd_dq_delay_timestep[env_ids] = 0
        if getattr(self.cfg.domain_rand, "randomize_ankle_pd_dq_filter", False):
            cutoff_range = getattr(self.cfg.domain_rand, "ankle_pd_dq_filter_cutoff_range", [8.0, 20.0])
            self.ankle_pd_dq_filter_alpha[env_ids] = self._sample_ankle_filter_alpha(env_ids, cutoff_range)
        else:
            self.ankle_pd_dq_filter_alpha[env_ids] = 1.0
        self.ankle_pd_dq_delay_buffer[env_ids] = 0.0
        self.ankle_pd_dq_filtered[env_ids] = 0.0
        self.ankle_pd_dq_filter_initialized[env_ids] = False

    def _delay_ankle_signal(self, signal, delay_buffer, delay_timestep):
        if delay_buffer.shape[-1] <= 1:
            delay_buffer[:, :, 0] = signal
            return signal
        delay_buffer[:, :, 1:] = delay_buffer[:, :, :-1].clone()
        delay_buffer[:, :, 0] = signal
        return delay_buffer[
            torch.arange(self.num_envs, device=self.device),
            :,
            delay_timestep.long(),
        ]

    def _filter_ankle_signal(self, signal, filtered, alpha, initialized):
        filtered[:] = torch.where(initialized, (1.0 - alpha) * filtered + alpha * signal, signal)
        initialized[:] = True
        return filtered

    def _get_ankle_dq_for_pd(self):
        dof_vel_for_pd = self.dof_vel.clone()
        ankle_dq = self.dof_vel[:, self._ankle_obs_joint_indices]
        if getattr(self.cfg.domain_rand, "randomize_ankle_pd_dq_delay", False):
            ankle_dq = self._delay_ankle_signal(
                ankle_dq,
                self.ankle_pd_dq_delay_buffer,
                self.ankle_pd_dq_delay_timestep,
            )
        if getattr(self.cfg.domain_rand, "randomize_ankle_pd_dq_filter", False):
            ankle_dq = self._filter_ankle_signal(
                ankle_dq,
                self.ankle_pd_dq_filtered,
                self.ankle_pd_dq_filter_alpha,
                self.ankle_pd_dq_filter_initialized,
            )
        if getattr(self.cfg.domain_rand, "randomize_ankle_pd_dq_noise", False):
            noise_std = float(getattr(self.cfg.domain_rand, "ankle_pd_dq_noise_std", 0.0))
            ankle_dq = ankle_dq + torch.randn_like(ankle_dq) * noise_std

        dof_vel_for_pd[:, self._ankle_obs_joint_indices] = ankle_dq
        return dof_vel_for_pd

    def _apply_actor_ankle_obs_bias(self, q, dq):
        """只给 actor 的关节位置/速度观测添加脚踝零偏，不改真实状态和 critic 观测。"""
        q_actor = q.clone()
        dq_actor = dq.clone()
        ankle_dq = dq_actor[:, self._ankle_obs_joint_indices]
        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_delay", False):
            ankle_dq = self._delay_ankle_signal(
                ankle_dq,
                self.ankle_obs_vel_delay_buffer,
                self.ankle_obs_vel_delay_timestep,
            )
        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_filter", False):
            ankle_dq = self._filter_ankle_signal(
                ankle_dq,
                self.ankle_obs_vel_filtered,
                self.ankle_obs_vel_filter_alpha,
                self.ankle_obs_vel_filter_initialized,
            )
        if getattr(self.cfg.domain_rand, "randomize_ankle_obs_vel_noise", False):
            noise_std = float(getattr(self.cfg.domain_rand, "ankle_obs_vel_noise_std", 0.0))
            ankle_dq = ankle_dq + torch.randn_like(ankle_dq) * noise_std * self.obs_scales.dof_vel

        q_actor[:, self._ankle_obs_joint_indices] += self.ankle_obs_pos_bias * self.obs_scales.dof_pos
        dq_actor[:, self._ankle_obs_joint_indices] = ankle_dq + self.ankle_obs_vel_bias * self.obs_scales.dof_vel
        return q_actor, dq_actor

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] += self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance_force[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance_force), space=gymapi.CoordinateSpace.LOCAL_SPACE)
        
    # def  _get_phase(self):
    #     cycle_time = self.cfg.rewards.cycle_time
    #     phase = self.episode_phase_buf * self.dt / cycle_time
    #     return phase

    # def _get_gait_phase(self):
    #     # return float mask 1 is stance, 0 is swing
    #     stance_mask = torch.zeros((self.num_envs, 2), device=self.device)

    #     foot_height = self.foot_height_buffer[self.ref_idx, self.phase_idx]
    #     stance_mask[foot_height < 0.07] = 1
    #     # stance_mask[torch.logical_or(self.phase_idx < 131, self.phase_idx > 209)] = 1
    #     return stance_mask

    def _init_reference_state_buffers(self):
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        self.ref_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.ref_pelvis_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_pelvis_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_pelvis_quat = self._identity_quat(self.num_envs, 1)
        self.ref_pelvis_ang_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_feet_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_feet_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_feet_quat = self._identity_quat(self.num_envs, 2)
        self.ref_feet_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_quat = self._identity_quat(self.num_envs, 2)
        self.ref_knee_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_quat = self._identity_quat(self.num_envs, 2)
        self.ref_hip_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_quat = self._identity_quat(self.num_envs, 2)
        self.ref_pelvic_yaw_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_waist_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_waist_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_waist_quat = self._identity_quat(self.num_envs, 1)
        self.ref_waist_ang_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_feet_contact = torch.zeros(self.num_envs, 2, device=self.device)
        self.ref_foot_height = torch.zeros(self.num_envs, 2, device=self.device)
        self.ref_root_linvel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_angvel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_euler_xyz = torch.zeros(self.num_envs, 3, device=self.device)

    def _identity_quat(self, num_envs, num_parts):
        quat = torch.zeros(num_envs, num_parts, 4, device=self.device)
        quat[..., 3] = 1.0
        return quat

    def _body_names_from_indices(self, indices):
        body_names = list(getattr(self, "body_names", []) or [])
        result = []
        for idx in indices.detach().cpu().tolist():
            idx = int(idx)
            result.append(body_names[idx] if 0 <= idx < len(body_names) else f"<body:{idx}>")
        return result

    def _resolve_tracking_body_indices(self, tracking_body_names=None):
        if not tracking_body_names:
            return torch.cat(
                (
                    self.base_indices,
                    self.feet_indices,
                    self.knee_indices,
                    self.hip_indices,
                    self.pelvic_yaw_indices,
                    self.waist_indices,
                ),
                dim=0,
            ).long()
        body_names = list(getattr(self, "body_names", []) or [])
        resolved = []
        for name in tracking_body_names:
            matches = [idx for idx, body_name in enumerate(body_names) if name in body_name]
            if not matches:
                raise ValueError(f"tracking_body_names entry '{name}' not found in asset bodies: {body_names}")
            if len(matches) > 1:
                matched_names = [body_names[idx] for idx in matches]
                raise ValueError(f"tracking_body_names entry '{name}' matched multiple bodies: {matched_names}")
            resolved.append(matches[0])
        indices = torch.tensor(resolved, dtype=torch.long, device=self.device)
        if len(torch.unique(indices)) != len(indices):
            raise ValueError(f"tracking_body_names resolved duplicate indices: {tracking_body_names}")
        return indices

    def _make_tracking_ref_specs(self, indices):
        source_groups = [
            ("pelvis", self.base_indices),
            ("feet", self.feet_indices),
            ("knee", self.knee_indices),
            ("hip", self.hip_indices),
            ("pelvic_yaw", self.pelvic_yaw_indices),
            ("waist", self.waist_indices),
        ]
        specs = []
        for body_id in indices.detach().cpu().tolist():
            body_id = int(body_id)
            found = None
            for kind, group_indices in source_groups:
                group = [int(idx) for idx in group_indices.detach().cpu().tolist()]
                if body_id in group:
                    found = (kind, group.index(body_id))
                    break
            if found is None:
                body_name = self._body_names_from_indices(torch.tensor([body_id], device=self.device))[0]
                raise ValueError(f"tracking body '{body_name}' is not mapped to a reference tensor.")
            specs.append(found)
        return specs

    def _init_tracking_reference_specs(self):
        self._tracking_ref_specs = self._make_tracking_ref_specs(self.all_tracking_indices)

    def _ref_tensors_for_kind(self, kind):
        if kind == "pelvis":
            return self.ref_pelvis_pos, self.ref_pelvis_quat, self.ref_pelvis_vel, self.ref_pelvis_ang_vel
        if kind == "feet":
            return self.ref_feet_pos, self.ref_feet_quat, self.ref_feet_vel, self.ref_feet_ang_vel
        if kind == "knee":
            return self.ref_knee_pos, self.ref_knee_quat, self.ref_knee_vel, self.ref_knee_ang_vel
        if kind == "hip":
            return self.ref_hip_pos, self.ref_hip_quat, self.ref_hip_vel, self.ref_hip_ang_vel
        if kind == "pelvic_yaw":
            return self.ref_pelvic_yaw_pos, self.ref_pelvic_yaw_quat, self.ref_pelvic_yaw_vel, self.ref_pelvic_yaw_ang_vel
        if kind == "waist":
            return self.ref_waist_pos, self.ref_waist_quat, self.ref_waist_vel, self.ref_waist_ang_vel
        raise KeyError(kind)

    def _tracking_ref_tensors(self):
        pos, quat, lin_vel, ang_vel = [], [], [], []
        for kind, part_idx in self._tracking_ref_specs:
            src_pos, src_quat, src_lin_vel, src_ang_vel = self._ref_tensors_for_kind(kind)
            pos.append(src_pos[:, part_idx : part_idx + 1])
            quat.append(src_quat[:, part_idx : part_idx + 1])
            lin_vel.append(src_lin_vel[:, part_idx : part_idx + 1])
            ang_vel.append(src_ang_vel[:, part_idx : part_idx + 1])
        return (
            torch.cat(pos, dim=1),
            torch.cat(quat, dim=1),
            torch.cat(lin_vel, dim=1),
            torch.cat(ang_vel, dim=1),
        )

    def _get_current_demo_lengths(self):
        return self.demo_lengths[self.ref_idx]

    def _get_future_phase_idx(self, time_idx_offset):
        max_phase = self._get_current_demo_lengths() - 1
        return torch.minimum(self.phase_idx + time_idx_offset, max_phase)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        n_ctrl = len(self.num_control)
        noise_vec[0:n_ctrl] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[n_ctrl:2 * n_ctrl] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[2 * n_ctrl:3 * n_ctrl] = 0.0  # actions
        noise_vec[3 * n_ctrl:3 * n_ctrl + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[3 * n_ctrl + 3:3 * n_ctrl + 6] = noise_scales.euler
        return noise_vec

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
        p_gains = self.p_gains * self.Kp_factors
        d_gains = self.d_gains * self.Kd_factors
        dof_vel_for_pd = self._get_ankle_dq_for_pd()

        if self.cfg.env.use_ref_actions:
            torques = p_gains * (actions - self.dof_pos) - d_gains * dof_vel_for_pd
        else:
            if self.cfg.domain_rand.action_delay:
                self.action_delay_buffer[:,:,1:] = self.action_delay_buffer[:,:,:-1].clone()
                self.action_delay_buffer[:,:,0] = actions_scaled.clone()
                delay_actions_scaled = self.action_delay_buffer[torch.arange(self.num_envs),:,self.action_delay_timestep.long()]
            else:
                delay_actions_scaled = actions_scaled

            # 非受控关节始终显式跟随参考轨迹，避免在 default_dof_pos != 0 时重复叠加默认角。
            target_dof = self.ref_dof_pos.clone()
            if getattr(self.cfg.control, "use_ref_residual_target", False):
                target_dof[:, self.num_control] = (
                    self.ref_dof_pos[:, self.num_control]
                    + delay_actions_scaled[:, self.num_control]
                )
            else:
                default_dof_pos_with_offset = self.default_dof_pos + self.default_dof_pos_offsets
                target_dof[:, self.num_control] = (
                    default_dof_pos_with_offset[:, self.num_control]
                    + delay_actions_scaled[:, self.num_control]
                )
            target_dof = target_dof + self.motor_offsets
            torques = p_gains * (target_dof - self.dof_pos) - d_gains * dof_vel_for_pd

        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strength_factors
        self.torques_raw = torques.clone()

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        if self.cfg.domain_rand.use_coulomb:
            friction_torques_left = self.cfg.domain_rand.left_Us * torch.tanh(self.dof_vel[:, [4,5]]/self.cfg.domain_rand.left_Qs) + self.cfg.domain_rand.left_Ud * self.dof_vel[:, [4,5]]
            friction_torques_right = self.cfg.domain_rand.right_Us * torch.tanh(self.dof_vel[:, [10,11]]/self.cfg.domain_rand.right_Qs) + self.cfg.domain_rand.right_Ud * self.dof_vel[:, [10,11]]

            torques[:, [4,5]] -= friction_torques_left
            torques[:, [10,11]] -= friction_torques_right

            # friction_torques_star = self.cfg.domain_rand.star_Us * torch.tanh(self.dof_vel[:, [0,1,2,3,6,7,8,9]]/self.cfg.domain_rand.star_Qs) + self.cfg.domain_rand.star_Ud * self.dof_vel[:, [0,1,2,3,6,7,8,9]]
            # torques[:, [0,1,2,3,6,7,8,9]] -= friction_torques_star
        return torques
    

    def compute_observations(self):
        self.compute_ref_state()
        norm_phase = self._get_reference_norm_phase()
        
        anchor_pos_w, anchor_quat_w = self._get_current_anchor_pose()
        anchor_pos_local, _ = self._get_current_anchor_pose_local()
        self.pelvis_p_b, self.pelvis_q_b = self.get_rel_pose(self.base_indices, anchor_pos_w, anchor_quat_w)
        self.feet_p_b, self.feet_q_b = self.get_rel_pose(self.feet_indices, anchor_pos_w, anchor_quat_w)
        self.knee_p_b, self.knee_q_b = self.get_rel_pose(self.knee_indices, anchor_pos_w, anchor_quat_w)
        self.hip_p_b, self.hip_q_b = self.get_rel_pose(self.hip_indices, anchor_pos_w, anchor_quat_w)
        self.pelvic_yaw_p_b, self.pelvic_yaw_q_b = self.get_rel_pose(self.pelvic_yaw_indices, anchor_pos_w, anchor_quat_w)
        self.waist_p_b, self.waist_q_b = self.get_rel_pose(self.waist_indices, anchor_pos_w, anchor_quat_w)
        self.pelvis_p0, self.pelvis_q0 = self.get_ref_rel_state_current(self.ref_pelvis_pos, self.ref_pelvis_quat)
        self.f_p0, self.f_q0 = self.get_ref_rel_state_current(self.ref_feet_pos, self.ref_feet_quat)
        self.k_p0, self.k_q0 = self.get_ref_rel_state_current(self.ref_knee_pos, self.ref_knee_quat)
        self.h_p0, self.h_q0 = self.get_ref_rel_state_current(self.ref_hip_pos, self.ref_hip_quat)
        self.pelvic_yaw_p0, self.pelvic_yaw_q0 = self.get_ref_rel_state_current(self.ref_pelvic_yaw_pos, self.ref_pelvic_yaw_quat)
        self.waist_p0, self.waist_q0 = self.get_ref_rel_state_current(self.ref_waist_pos, self.ref_waist_quat)
        tracking_ref_pos, tracking_ref_quat, _, _ = self._tracking_ref_tensors()
        tracking_p_b, tracking_q_b = self.get_rel_pose(self.all_tracking_indices, anchor_pos_w, anchor_quat_w)
        tracking_p0, tracking_q0 = self.get_ref_rel_state_current(tracking_ref_pos, tracking_ref_quat)
        
        # A. 当前帧 (t)
        # Actor q uses current controlled joint positions.  The reference joint
        # position and velocity are provided in goal_buf.
        default_dof_pos_with_offset = self.default_dof_pos + self.default_dof_pos_offsets
        ref_dof_pos_abs_curr = self.ref_dof_pos
        ref_dof_pos_curr = ref_dof_pos_abs_curr[:, self.num_control] * self.obs_scales.dof_pos
        ref_dof_vel_curr = self.ref_dof_vel[:, self.num_control] * self.obs_scales.dof_vel


        # 计算位置误差 (在机器人局部坐标系下)
        ref_anchor_pos = self.ref_waist_pos[:, 0, :]
        ref_anchor_quat = self.ref_waist_quat[:, 0, :]
        # delta_pos = R_robot^T * (p_ref - p_robot)
        anchor_pos_b = quat_rotate_inverse(anchor_quat_w, ref_anchor_pos - anchor_pos_local)

        # 计算方向误差 (在机器人局部坐标系下)
        # delta_quat = R_robot^T * R_ref
        # 注意：这里需要四元数共轭乘法
        anchor_quat_b = quat_mul(quat_conjugate(anchor_quat_w), ref_anchor_quat)
        
        # 将四元数转换为旋转矩阵的前两列 (6D 连续表示)，这是 Beyond Mimic 的做法
        anchor_ori_mat = self.matrix_from_quat(anchor_quat_b) # [N, 3, 3]
        anchor_ori_b = anchor_ori_mat[..., :2].reshape(self.num_envs, -1) # [N, 6]
        # -------------------------
        # Body tracking error (local frame)
        # -------------------------
        pelvis_err_p = self.pelvis_p_b - self.pelvis_p0
        feet_err_p = self.feet_p_b - self.f_p0
        knee_err_p = self.knee_p_b - self.k_p0
        hip_err_p = self.hip_p_b - self.h_p0
        pelvic_yaw_err_p = self.pelvic_yaw_p_b - self.pelvic_yaw_p0
        waist_err_p = self.waist_p_b - self.waist_p0

        pelvis_err_q = self._quat_err_6d(self.pelvis_q_b, self.pelvis_q0, 1)  # 6
        feet_err_q = self._quat_err_6d(self.feet_q_b, self.f_q0, 2)  # 12
        knee_err_q = self._quat_err_6d(self.knee_q_b, self.k_q0, 2)  # 12
        hip_err_q = self._quat_err_6d(self.hip_q_b, self.h_q0, 2)  # 12
        pelvic_yaw_err_q = self._quat_err_6d(self.pelvic_yaw_q_b, self.pelvic_yaw_q0, 2)  # 12
        waist_err_q = self._quat_err_6d(self.waist_q_b, self.waist_q0, 1)  # 6
        
        # self.privileged_obs_buf = torch.cat((
        #     self.root_states[:, 2:3],  # 1
        #     (self.dof_pos - self.default_joint_pd_target)[:, self.num_control] * self.obs_scales.dof_pos,  # 12 13
        #     (self.dof_vel[:, self.num_control] * self.obs_scales.dof_vel),  # 12 25
        #     self.actions[:, self.num_control],  # 12 37
        #     self.base_lin_vel * self.obs_scales.lin_vel,  # 3  40
        #     self.base_ang_vel * self.obs_scales.ang_vel,  # 3  43
        #     self.projected_gravity, # 3 46
        #     # root_pos, root_quat,  # 7 53
        #     self.feet_p_b, self.feet_q_b, # 14  67
        #     self.knee_p_b, self.knee_q_b,   #14  81
        #     self.hip_p_b, self.hip_q_b,   # 14  95
        #     self.head_p_b, self.head_q_b, # 7  102
        #     self.waist_p_b, self.waist_q_b,
        #     self.rand_push_force[:, :2] / self.cfg.domain_rand.max_push_vel_xy,  # 2  104
        #     self.rand_push_torque / self.cfg.domain_rand.max_push_ang_vel,  # 3 107
        #     self.disturbance_force[:, 0, :] / self.cfg.domain_rand.disturbance_range[1],  # 3  110
        #     (self.friction_coeffs - self.cfg.domain_rand.friction_range[0])/(self.cfg.domain_rand.friction_range[1]-self.cfg.domain_rand.friction_range[0]),  # 1  111
        #     (self.restitution_coeffs - self.cfg.domain_rand.restitution_range[0])/(self.cfg.domain_rand.restitution_range[1]-self.cfg.domain_rand.restitution_range[0]),  # 1  112
        #     (self.Kp_factors[:, self.num_control] - self.cfg.domain_rand.kp_range[0])/(self.cfg.domain_rand.kp_range[1]-self.cfg.domain_rand.kp_range[0]),  #  12  124
        #     (self.Kd_factors[:, self.num_control] - self.cfg.domain_rand.kd_range[0])/(self.cfg.domain_rand.kd_range[1]-self.cfg.domain_rand.kd_range[0]),  #  12  136
        #     (self.payload - self.cfg.domain_rand.payload_mass_range[0])/(self.cfg.domain_rand.payload_mass_range[1]-self.cfg.domain_rand.payload_mass_range[0]),  # 1 137
        #     # (self.com_displacement - self.cfg.domain_rand.com_displacement_range[0])/(self.cfg.domain_rand.com_displacement_range[1]-self.cfg.domain_rand.com_displacement_range[0]), # 3 140
        #      self.com_displacement * self.obs_scales.com_pos,
        #     norm_phase, # 1  141
        #     anchor_pos_b,       # 3维
        #     anchor_ori_b,       # 6维
        # ), dim=-1)  #127  /  134
        
        base_rp = get_euler_xyz_tensor(self.root_states[:, 3:7])[:, 0:2]  # roll, pitch
        q_priv_hist = (self.dof_pos - default_dof_pos_with_offset) * self.obs_scales.dof_pos
        dq_priv_hist = self.dof_vel * self.obs_scales.dof_vel
        priv_obs_hist_part = torch.cat((
            self.root_states[:, 2:3],                                                                          # 机身高度 z: 1
            base_rp,                                                                                            # 机身姿态 roll+pitch: 2
            q_priv_hist[:, self.num_control],                                                                  # 12
            dq_priv_hist[:, self.num_control],                                                                 # 12
            self.actions[:, self.num_control],                                                                  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,                                                       # 3
            self.base_ang_vel * self.obs_scales.ang_vel,                                                       # 3
        ), dim=-1)   # [N, 45]
        dif_root_linvel = (self.base_lin_vel - self.ref_root_linvel) * self.obs_scales.lin_vel
        dif_root_angvel = (self.base_ang_vel - self.ref_root_angvel) * self.obs_scales.ang_vel
        # 参考脚抬脚状态（1=抬脚，0=不抬脚），仅特权观测。feet_contact_buffer 存接触掩码(1=触地)，取反得抬脚
        ref_foot_contact_curr = 1.0 - self.ref_feet_contact
        
        priv_curr_dim = int(
            getattr(self.cfg.env, "num_privileged_obs", 45 + 146 + 31)
            - getattr(self.cfg.env, "single_num_privileged_obs", 45)
            - getattr(self.cfg.env, "num_goal_obs", 31)
        )
        if priv_curr_dim == 119:
            num_tracking_parts = len(self.all_tracking_indices)
            tracking_err_p = tracking_p_b - tracking_p0
            tracking_err_q = self._quat_err_6d(tracking_q_b, tracking_q0, num_tracking_parts)
            priv_obs_curr_part = torch.cat((
                anchor_pos_b,   # 3
                anchor_ori_b,   # 6
                dif_root_linvel,  # 3
                dif_root_angvel,  # 3
                tracking_err_p,   # 7*3
                tracking_err_q,   # 7*6
                self.rand_push_force[:, :2] / self.cfg.domain_rand.max_push_vel_xy,  # 2
                self.rand_push_torque / self.cfg.domain_rand.max_push_ang_vel,  # 3
                self.disturbance_force[:, 0, :] / self.cfg.domain_rand.disturbance_range[1],  # 3
                (self.friction_coeffs - self.cfg.domain_rand.friction_range[0]) / (self.cfg.domain_rand.friction_range[1] - self.cfg.domain_rand.friction_range[0]), # 1
                (self.restitution_coeffs - self.cfg.domain_rand.restitution_range[0]) / (self.cfg.domain_rand.restitution_range[1] - self.cfg.domain_rand.restitution_range[0]), # 1
                (self.Kp_factors[:, self.num_control] - self.cfg.domain_rand.kp_range[0]) / (self.cfg.domain_rand.kp_range[1] - self.cfg.domain_rand.kp_range[0]),  # 12
                (self.Kd_factors[:, self.num_control] - self.cfg.domain_rand.kd_range[0]) / (self.cfg.domain_rand.kd_range[1] - self.cfg.domain_rand.kd_range[0]), # 12
                (self.payload - self.cfg.domain_rand.payload_mass_range[0]) / (self.cfg.domain_rand.payload_mass_range[1] - self.cfg.domain_rand.payload_mass_range[0]), # 1
                self.com_displacement * self.obs_scales.com_pos, # 3
                ref_foot_contact_curr,  # 2：左/右脚参考抬脚（1=抬脚，0=不抬脚）
                self._get_privileged_reference_phase_obs(norm_phase), # 1
            ), dim=-1)
        else:
            priv_obs_curr_part = torch.cat((
                anchor_pos_b,   # 3
                anchor_ori_b,   # 6
                dif_root_linvel,  # 3：根线速度误差
                dif_root_angvel,  # 3：根角速度误差
                pelvis_err_p,   # 3：骨盆位置误差
                pelvis_err_q,   # 6：骨盆姿态误差（6D）
                feet_err_p,     # 6：双脚位置误差
                feet_err_q,     # 12：双脚姿态误差（6D×2）
                knee_err_p,     # 6：双膝位置误差
                knee_err_q,     # 12：双膝姿态误差（6D×2）
                hip_err_p,      # 6：双髋位置误差
                hip_err_q,      # 12：双髋姿态误差（6D×2）
                pelvic_yaw_err_p,  # 6：双侧 pelvic_yaw 位置误差
                pelvic_yaw_err_q,  # 12：双侧 pelvic_yaw 姿态误差（6D×2）
                waist_err_p,    # 3：腰位置误差
                waist_err_q,    # 6：腰姿态误差（6D）
                self.rand_push_force[:, :2] / self.cfg.domain_rand.max_push_vel_xy,  # 2
                self.rand_push_torque / self.cfg.domain_rand.max_push_ang_vel,  # 3
                self.disturbance_force[:, 0, :] / self.cfg.domain_rand.disturbance_range[1],  # 3
                (self.friction_coeffs - self.cfg.domain_rand.friction_range[0]) / (self.cfg.domain_rand.friction_range[1] - self.cfg.domain_rand.friction_range[0]), # 1
                (self.restitution_coeffs - self.cfg.domain_rand.restitution_range[0]) / (self.cfg.domain_rand.restitution_range[1] - self.cfg.domain_rand.restitution_range[0]), # 1
                (self.Kp_factors[:, self.num_control] - self.cfg.domain_rand.kp_range[0]) / (self.cfg.domain_rand.kp_range[1] - self.cfg.domain_rand.kp_range[0]),  # 12
                (self.Kd_factors[:, self.num_control] - self.cfg.domain_rand.kd_range[0]) / (self.cfg.domain_rand.kd_range[1] - self.cfg.domain_rand.kd_range[0]), # 12
                (self.payload - self.cfg.domain_rand.payload_mass_range[0]) / (self.cfg.domain_rand.payload_mass_range[1] - self.cfg.domain_rand.payload_mass_range[0]), # 1
                self.com_displacement * self.obs_scales.com_pos, # 3
                ref_foot_contact_curr,  # 2：左/右脚参考抬脚（1=抬脚，0=不抬脚）
                self._get_privileged_reference_phase_obs(norm_phase), # 1
            ), dim=-1)  # 146
        if priv_obs_curr_part.shape[1] != priv_curr_dim:
            raise RuntimeError(f"privileged current dim mismatch: got {priv_obs_curr_part.shape[1]}, cfg={priv_curr_dim}")
        
        if self.cfg.domain_rand.sys_delay:
            root_states_ = self.obs_imu_delay_buffer[torch.arange(self.num_envs), :, self.obs_imu_delay_timestep.long()]
            dof_pos_vel_ = self.obs_motor_delay_buffer[torch.arange(self.num_envs), :, self.obs_motor_delay_timestep.long()]

            q = (dof_pos_vel_[:, :self.num_actions] - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = dof_pos_vel_[:, self.num_actions:] * self.obs_scales.dof_vel

            base_ang_vel_ = quat_rotate_inverse(root_states_[:, 3:7], root_states_[:, 10:13])
            base_euler_xyz = get_euler_xyz_tensor(root_states_[:, 3:7])[:, 0:3]

        else:
            q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = self.dof_vel * self.obs_scales.dof_vel
            base_ang_vel_ = self.base_ang_vel
            base_euler_xyz = self.base_euler_xyz[:, 0:3]

        q, dq = self._apply_actor_ankle_obs_bias(q, dq)

        obs_euler_xyz = base_euler_xyz.clone()
        obs_euler_xyz[:, 2] = wrap_to_pi(obs_euler_xyz[:, 2] - self.initial_base_yaw)

        obs_buf = torch.cat((
            q[:, self.num_control],    # 12D
            dq[:, self.num_control],  # 12D
            self.actions[:, self.num_control],   # 12D
            base_ang_vel_ * self.obs_scales.ang_vel,  # 3
            obs_euler_xyz * self.obs_scales.quat,  # 3
            self._get_actor_reference_extra_obs(),
        ), dim=-1)

        

        # --- 6. 构建 Goal Buffer ---
        ref_waist_pos = self.ref_waist_pos[:, 0, :]
        ref_waist_quat = self.ref_waist_quat[:, 0, :]
        ref_waist_euler = get_euler_xyz_tensor(ref_waist_quat)  # roll, pitch, yaw
        ref_waist_vel = self.ref_waist_vel[:, 0, :]
        ref_waist_ang_vel = self.ref_waist_ang_vel[:, 0, :]
        goal_terms = [
            ref_dof_pos_curr,                                          # 12：当前帧参考腿部关节位置（仅 num_control）
            ref_dof_vel_curr,                                          # 12：当前帧参考腿部关节速度（仅 num_control）
            ref_waist_pos[:, 2:3],                                     # 1：参考 waist 高度 z
            ref_waist_euler[:, 0:2],                                   # 2：参考 waist roll+pitch
            ref_waist_vel * self.obs_scales.lin_vel,                   # 3：参考 waist 线速度
            ref_waist_ang_vel[:, 2:3] * self.obs_scales.ang_vel,       # 1：参考 waist 角速度 z 轴
        ]
        if int(getattr(self.cfg.env, "num_goal_obs", 31)) >= 33:
            goal_terms.append(self.ref_feet_contact.float())           # 2：参考脚底接触，1=触地，0=离地
        goal_buf = torch.cat(goal_terms, dim=-1)
        
        
        '''
        goal_buf = torch.cat((
            self.root_states_buffer[self.ref_idx, self.phase_idx][:, 2:3], # ref_root_height  3
            self.root_states_buffer[self.ref_idx, self.phase_idx][:, 3:7], # ref_root_quat  4  7
            # self.root_linvel_buffer[self.ref_idx, self.phase_idx],
            # self.root_linvel_buffer[self.ref_idx, self.phase_idx+10],
            # self.root_linvel_buffer[self.ref_idx, self.phase_idx+19],  
            # self.root_angvel_buffer[self.ref_idx, self.phase_idx],
            # self.root_angvel_buffer[self.ref_idx, self.phase_idx+10],
            # self.root_angvel_buffer[self.ref_idx, self.phase_idx+19],  
            # self.euler_xyz_buffer[self.ref_idx, self.phase_idx][:, 0:2],
            # self.euler_xyz_buffer[self.ref_idx, self.phase_idx+10][:, 0:2],
            # self.euler_xyz_buffer[self.ref_idx, self.phase_idx+19][:, 0:2], 
            # self.euler_z_offset[:, 0:1], # 1 
            # self.dof_pos_buffer[self.ref_idx, self.phase_idx][:, 0:13], # 12  19
            self.dof_pos_buffer[self.ref_idx, self.phase_idx],
            # self.dof_pos_buffer[self.ref_idx, self.phase_idx+10][:, 0:12],
            # self.dof_pos_buffer[self.ref_idx, self.phase_idx+19][:, 0:12],  
            # self.dof_vel_buffer[self.ref_idx, self.phase_idx][:, 0:13],   # 12  31
            self.pelvis_p0, self.pelvis_q0, self.f_p0, self.f_q0, self.k_p0, self.k_q0, self.h_p0, self.h_q0, # 49维
            self.waist_p0, self.waist_q0,
            self.pelvic_yaw_p0, self.pelvic_yaw_q0,
            self.pelvis_p10, self.pelvis_q10, self.f_p10, self.f_q10, self.k_p10, self.k_q10, self.h_p10, self.h_q10,
            self.waist_p10, self.waist_q10,
            self.pelvic_yaw_p10, self.pelvic_yaw_q10,
        ), dim=-1)
        '''

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        if self.add_noise:  
            #obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_buf = torch.cat((obs_now, goal_buf), dim=-1)
        # privileged_obs_buf_ = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        # privileged_obs_buf_ = torch.cat((
        #     privileged_obs_buf_,
        #     goal_buf,
        # ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            priv_obs_hist_part,  # 45
            priv_obs_curr_part,
            goal_buf,
        ), dim=-1)
        if not getattr(self, "_printed_observation_layout_shapes", False):
            print(
                "[mrobot_mimic][IsaacGym] Observation layout changed: Dance actor obs 61/73 -> 75 "
                "or BPM actor obs 64 -> 76. "
                "Old checkpoints and normalizer statistics are incompatible. "
                "Train from scratch or reset normalizer.",
                flush=True,
            )
            print(
                "[mrobot_mimic][IsaacGym] observation shapes: "
                f"obs_now.shape={tuple(obs_now.shape)}, goal_buf.shape={tuple(goal_buf.shape)}, "
                f"actor obs.shape={tuple(self.obs_buf.shape)}, privileged obs.shape={tuple(self.privileged_obs_buf.shape)}",
                flush=True,
            )
            self._printed_observation_layout_shapes = True
        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(
                f"obs dim mismatch: got {self.obs_buf.shape[1]}, cfg={self.cfg.env.num_observations}"
            )
        if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"privileged obs dim mismatch: got {self.privileged_obs_buf.shape[1]}, "
                f"cfg={self.cfg.env.num_privileged_obs}"
            )
        # print("ENV OBS[0]:", obs_buf[0, :45]) # 打印前45维看看
        # print("ENV GOAL[0]:", goal_buf[0, :10])    # 打印 Goal 前10维


    def computer_aux(self):
        # contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        # q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        # dq = self.dof_vel * self.obs_scales.dof_vel

        # self.phase_idx = self.episode_phase_buf % self.demo_length
        # pos_target = self.dof_pos_buffer[self.phase_idx]
        # dof_vel_target = self.dof_vel_buffer[self.phase_idx]

        # self.aux = torch.cat((
        #     q[:,self.num_control],  # 12
        #     dq[:,self.num_control],  # 12
        #     self.base_ang_vel * self.obs_scales.ang_vel,  # 3
        #     self.base_lin_vel * self.obs_scales.lin_vel,  # 3
        #     self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
        #     self.root_states[:, 2:3],  # 1
        #     # contact_mask,
        # ), dim=-1)

        self.aux = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.root_states[:, 2:3],  # 1
            (self.friction_coeffs - self.cfg.domain_rand.friction_range[0])/(self.cfg.domain_rand.friction_range[1]-self.cfg.domain_rand.friction_range[0]),  # 1
            (self.payload - self.cfg.domain_rand.payload_mass_range[0])/(self.cfg.domain_rand.payload_mass_range[1]-self.cfg.domain_rand.payload_mass_range[0]),  # 1
            self.com_displacement * self.obs_scales.com_pos,  # 3
        ), dim=-1)  # 9

    def _get_current_anchor_pose(self):
        waist_idx = int(self.waist_indices[0].item())
        return self.rigid_state[:, waist_idx, :3], self.rigid_state[:, waist_idx, 3:7]

    def _get_current_anchor_pose_local(self):
        anchor_pos_w, anchor_quat = self._get_current_anchor_pose()
        return anchor_pos_w - self.env_origins, anchor_quat

    def _get_ref_anchor_pose(self, target_idx):
        return (
            self.ref_waist_pos[:, 0, :],
            self.ref_waist_quat[:, 0, :],
        )

    def _get_anchor_yaw_alignment(self, target_idx, cur_anchor_quat=None):
        if cur_anchor_quat is None:
            _, cur_anchor_quat = self._get_current_anchor_pose()
        _, ref_anchor_quat = self._get_ref_anchor_pose(target_idx)
        q_diff = quat_mul(cur_anchor_quat, quat_inv(ref_anchor_quat))
        return torch_utils.calc_heading_quat(q_diff)

    def _align_ref_positions_to_current_anchor(self, ref_body_pos, target_idx, cur_anchor_pos=None, cur_anchor_quat=None):
        if cur_anchor_pos is None or cur_anchor_quat is None:
            cur_anchor_pos, cur_anchor_quat = self._get_current_anchor_pose_local()
        ref_anchor_pos, _ = self._get_ref_anchor_pose(target_idx)
        yaw_diff_quat = self._get_anchor_yaw_alignment(target_idx, cur_anchor_quat)
        rel_ref_pos = ref_body_pos - ref_anchor_pos.unsqueeze(1)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_pos.shape[1], -1)
        rotated_rel_ref = quat_apply(yaw_diff_repeat, rel_ref_pos)
        target_pos = cur_anchor_pos.unsqueeze(1) + rotated_rel_ref
        target_pos[:, :, 2] = ref_body_pos[:, :, 2]
        return target_pos

    def _align_ref_quats_to_current_anchor(self, ref_body_quat, target_idx, cur_anchor_quat=None):
        yaw_diff_quat = self._get_anchor_yaw_alignment(target_idx, cur_anchor_quat)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_quat.shape[1], -1)
        return torch_utils.quat_mul(yaw_diff_repeat, ref_body_quat)

    def _align_ref_vectors_to_current_anchor(self, ref_body_vec, target_idx, cur_anchor_quat=None):
        yaw_diff_quat = self._get_anchor_yaw_alignment(target_idx, cur_anchor_quat)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_vec.shape[1], -1)
        return quat_apply(yaw_diff_repeat, ref_body_vec)

    def get_ref_rel_state_current(self, ref_pos_w, ref_quat_w):
        r_pos_w = self.ref_waist_pos[:, 0, :]
        r_quat_w = self.ref_waist_quat[:, 0, :]
        r_inv_quat = quat_inv(r_quat_w)
        num_sub_keys = ref_pos_w.shape[1]
        diff_p = ref_pos_w - r_pos_w.unsqueeze(1)
        r_inv_quat_expanded = r_inv_quat.unsqueeze(1).expand(-1, num_sub_keys, -1)
        rel_p = quat_apply(r_inv_quat_expanded.reshape(-1, 4), diff_p.reshape(-1, 3)).reshape(
            self.num_envs, num_sub_keys, 3
        )
        r_inv_quat_expanded = r_inv_quat.unsqueeze(1).expand(-1, num_sub_keys, -1)
        rel_q = quat_mul(
            r_inv_quat_expanded.reshape(-1, 4), ref_quat_w.reshape(-1, 4)
        ).reshape(self.num_envs, num_sub_keys, 4)
        return rel_p.reshape(self.num_envs, -1), rel_q.reshape(self.num_envs, -1)

    # 计算未来多帧关键点机体系相对位姿
    def get_ref_rel_state_step(self, time_idx_offset, p_buffer, q_buffer):
        target_idx = self._get_future_phase_idx(time_idx_offset)
        r_pos_w, r_quat_w = self._get_ref_anchor_pose(target_idx)
        r_inv_quat = quat_inv(r_quat_w)
        k_pos_w = p_buffer[self.ref_idx, target_idx] # [num_envs, num_sub_keys, 3]
        k_quat_w = q_buffer[self.ref_idx, target_idx] # [num_envs, num_sub_keys, 4]
        num_sub_keys = k_pos_w.shape[1] 
        diff_p = k_pos_w - r_pos_w.unsqueeze(1)
        r_quat_repeat = r_quat_w.unsqueeze(1).repeat(1, num_sub_keys, 1).reshape(-1, 4)
        r_inv_quat_repeat = r_inv_quat.unsqueeze(1).repeat(1, num_sub_keys, 1).reshape(-1, 4)
        rel_p_b = quat_rotate_inverse(r_quat_repeat, diff_p.reshape(-1, 3)).reshape(self.num_envs, -1)
        rel_q_b = quat_mul(r_inv_quat_repeat, k_quat_w.reshape(-1, 4)).reshape(self.num_envs, -1)
        
        return rel_p_b, rel_q_b
    
    def get_rel_pose(self, indices, root_pos, root_quat):
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        
        pos_w = self.rigid_state[:, indices, :3]
        quat_w = self.rigid_state[:, indices, 3:7]
        num_parts = indices.shape[0]
        
        p_rel_w = pos_w - root_pos.unsqueeze(1)
        
        r_quat_repeat = root_quat.unsqueeze(1).repeat(1, num_parts, 1).reshape(-1, 4)
        
        p_rel_b = quat_rotate_inverse(r_quat_repeat, p_rel_w.reshape(-1, 3))
        p_rel_b = p_rel_b.reshape(self.num_envs, num_parts * 3)
        
        inv_root_quat = quat_inv(root_quat)
        inv_root_repeat = inv_root_quat.unsqueeze(1).repeat(1, num_parts, 1).reshape(-1, 4)
        q_rel_b = quat_mul(inv_root_repeat, quat_w.reshape(-1, 4))
        q_rel_b = q_rel_b.reshape(self.num_envs, num_parts * 4)
        
        return p_rel_b, q_rel_b

    def _get_aligned_body_pos_targets(self, indices, ref_body_pos):
        cur_body_pos = self.rigid_state[:, indices, :3] - self.env_origins.unsqueeze(1)
        cur_anchor_pos, cur_anchor_quat = self._get_current_anchor_pose_local()
        target_pos = self._align_ref_positions_to_current_anchor(
            ref_body_pos,
            self.phase_idx,
            cur_anchor_pos,
            cur_anchor_quat,
        )
        return cur_body_pos, target_pos

    def _get_aligned_body_quat_targets(self, indices, ref_body_quat):
        cur_body_quat = self.rigid_state[:, indices, 3:7]
        _, cur_anchor_quat = self._get_current_anchor_pose()
        target_quat = self._align_ref_quats_to_current_anchor(ref_body_quat, self.phase_idx, cur_anchor_quat)
        return cur_body_quat, target_quat

    def _get_aligned_body_vector_targets(self, indices, ref_body_vec, state_slice):
        cur_body_vec = self.rigid_state[:, indices, state_slice]
        _, cur_anchor_quat = self._get_current_anchor_pose()
        target_vec = self._align_ref_vectors_to_current_anchor(ref_body_vec, self.phase_idx, cur_anchor_quat)
        return cur_body_vec, target_vec
    
    def matrix_from_quat(self, quat):
        """
        Convert a quaternion to a rotation matrix.
        Args:
            quat: (..., 4) tensor [x, y, z, w]
        Returns:
            rot_mat: (..., 3, 3) tensor
        """
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        
        row0 = torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row1 = torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1)
        row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1)
        
        return torch.stack([row0, row1, row2], dim=-2)

    def _quat_err_6d(self, q_curr_flat, q_ref_flat, num_parts):
        """计算姿态误差并转为 6D 连续旋转表示。
        q_curr_flat / q_ref_flat: [N, num_parts*4]
        返回: [N, num_parts*6]
        """
        q_c = q_curr_flat.reshape(-1, 4)
        q_r = q_ref_flat.reshape(-1, 4)
        err_q = quat_mul(quat_conjugate(q_c), q_r)
        err_mat = self.matrix_from_quat(err_q)           # [-1, 3, 3]
        err_6d = err_mat[..., :2].reshape(-1, 6)         # [-1, 6]
        return err_6d.reshape(self.num_envs, num_parts * 6)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.curriculum_episode_length_buf += 1
        self.episode_phase_buf += 1
        self.common_step_counter += 1
        self._advance_reference_phase()

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.computer_aux()
        if self.is_amp:
            self.computer_disc_obs()
        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_torques = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        termination_contact_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.base_too_low_buf = self.root_states[:, 2] < 0.5
        grace_mask = self.episode_length_buf > 5
        self.fall_reset_buf = (termination_contact_buf | self.base_too_low_buf) & grace_mask
        self.ref_end_reset_buf[:] = False
        self.reset_buf = self.fall_reset_buf | self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.compute_ref_state()
        env_dof_pos = self.ref_dof_pos[env_ids].clone()
        env_dof_vel = self.ref_dof_vel[env_ids].clone()
        if getattr(self.cfg.domain_rand, "randomize_init_dof_pos", False):
            init_dof_pos_range = getattr(self.cfg.domain_rand, "init_dof_pos_range", [-0.05, 0.05])
            env_dof_pos += torch_rand_float(
                init_dof_pos_range[0],
                init_dof_pos_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        self.dof_pos[env_ids] = env_dof_pos
        self.dof_vel[env_ids] = env_dof_vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        self.root_states[env_ids, 7:9] = torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
        if getattr(self.cfg.domain_rand, "randomize_root_xy_reset", False):
            xy_range = getattr(self.cfg.domain_rand, "root_xy_reset_range", [0.0, 0.0])
            self.root_states[env_ids, 0:2] += torch_rand_float(
                xy_range[0], xy_range[1], (len(env_ids), 2), device=self.device
            )
        if getattr(self.cfg.domain_rand, "randomize_root_yaw_reset", False):
            yaw_range = getattr(self.cfg.domain_rand, "root_yaw_reset_range", [0.0, 0.0])
            yaw_noise = torch_rand_float(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=self.device)[:, 0]
            zero_tensor = torch.zeros_like(yaw_noise)
            yaw_noise_quat = quat_from_euler_xyz(zero_tensor, zero_tensor, yaw_noise)
            self.root_states[env_ids, 3:7] = quat_mul(yaw_noise_quat, self.root_states[env_ids, 3:7])
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        self._resample_reference_commands(env_ids)
        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.refresh_actor_rigid_shape_props(env_ids)

        self._resample_ankle_obs_bias(env_ids)
        self._resample_ankle_dq_randomization(env_ids)
        self.initial_base_yaw[env_ids] = get_euler_xyz_tensor(self.root_states[env_ids, 3:7])[:, 2]

        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] / (self.episode_length_buf[env_ids] + 1.)
            )
            self.episode_sums[key][env_ids] = 0.
        if hasattr(self, "tracking_score_sums"):
            for key in self.tracking_score_sums.keys():
                self.extras["episode"]['score_' + key] = torch.mean(
                    self.tracking_score_sums[key][env_ids] / (self.episode_length_buf[env_ids] + 1.)
                )
                self.tracking_score_sums[key][env_ids] = 0.
        self.extras["episode"]["mean_episode_length"] = torch.mean(self.curriculum_episode_length_buf[env_ids].float())
        self.extras["episode"]["fall_ratio"] = torch.mean(self.fall_reset_buf[env_ids].float())
        self.extras["episode"]["ref_end_ratio"] = torch.zeros((), device=self.device)
        self.extras["episode"]["time_out_ratio"] = torch.mean(self.time_out_buf[env_ids].float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.curriculum_episode_length_buf[env_ids] = 0
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.last_root_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.last_root_offset[env_ids] = 0
        self.last_landing_contacts[env_ids] = False
        self.last_landing_contacts_filt[env_ids] = False
    
# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 20.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_stance_foot_slip(self):
        """惩罚支撑脚脚底滑动：带速度死区，重点抑制双脚支撑时的横向滑步。"""
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        foot_vel_xy = self.rigid_state[:, self.feet_indices, 7:9]
        foot_speed_xy = torch.norm(foot_vel_xy, dim=2)
        stance_threshold = 20.0
        stance_mask = (contact_force_z > stance_threshold).float()
        # 小速度常是接触噪声和求解误差，不应过罚。
        speed_deadzone = 0.03
        slip_excess = torch.clamp(foot_speed_xy - speed_deadzone, min=0.0)
        slip_per_foot = torch.square(slip_excess) * stance_mask
        # 双脚同时支撑时额外抑制“站立滑步”。
        double_stance = (torch.sum(stance_mask, dim=1, keepdim=True) >= 2.0).float()
        slip_per_foot *= (1.0 + double_stance)
        slip_amount = torch.sum(slip_per_foot, dim=1)
        return slip_amount    

    def _reward_stance_feet_speed_reg(self):
        """仅惩罚支撑脚绕 yaw 方向角速度过大，抑制脚底在地上打转。"""
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        stance_mask = (contact_force_z > 20.0).float()

        foot_yaw_ang_vel = torch.abs(self.rigid_state[:, self.feet_indices, 12])

        # 小范围 yaw 角速度常来自接触抖动与求解器噪声，留一点死区避免过罚。
        yaw_deadzone = 0.5
        yaw_excess = torch.clamp(foot_yaw_ang_vel - yaw_deadzone, min=0.0)

        per_foot_penalty = torch.square(yaw_excess) * stance_mask
        return torch.sum(per_foot_penalty, dim=1)

    def _reward_landing_foot_z_vel(self):
        """惩罚摆动脚第一次落地瞬间的竖直速度过大，鼓励更柔和地落脚。"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        contact_filt = torch.logical_or(contact, self.last_landing_contacts)
        from_air_to_ground = torch.logical_and(contact_filt, ~self.last_landing_contacts_filt)
        self.last_landing_contacts = contact
        self.last_landing_contacts_filt = contact_filt

        foot_vel_z = torch.abs(self.rigid_state[:, self.feet_indices, 9])
        z_vel_deadzone = 0.12
        z_vel_excess = torch.clamp(foot_vel_z - z_vel_deadzone, min=0.0)
        landing_penalty = torch.square(z_vel_excess) * from_air_to_ground.float()
        return torch.sum(landing_penalty, dim=1)

    def _reward_pre_landing_foot_z_vel(self):
        """惩罚摆动脚近地但未接触时的向下速度，避免脚从空中砸地。"""
        ref_contact = self.ref_feet_contact.float()
        ref_swing = 1.0 - ref_contact
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        no_contact = (contact_force_z < 5.0).float()
        foot_z = self.rigid_state[:, self.feet_indices, 2]
        near_ground = ((foot_z > 0.055) & (foot_z < 0.12)).float()

        foot_vel_z = self.rigid_state[:, self.feet_indices, 9]
        downward_vel_excess = torch.clamp(-foot_vel_z - 0.15, min=0.0)
        pre_landing_mask = ref_swing * no_contact * near_ground
        return torch.sum(torch.square(downward_vel_excess) * pre_landing_mask, dim=1)

    def _reward_pre_landing_foot_smooth(self):
        """落地前近地阶段惩罚足底 roll/pitch 快速翻转和姿态误差，鼓励提前调平。"""
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        foot_z = self.rigid_state[:, self.feet_indices, 2]
        pre_landing_mask = (
            (contact_force_z < 5.0)
            & (foot_z > 0.06)
            & (foot_z < 0.10)
        ).float()

        # 落地前脚底 roll/pitch 角速度不要太大，避免快触地时还在翻脚。
        foot_rp_ang_vel = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rp_ang_vel_excess = torch.clamp(foot_rp_ang_vel - 0.5, min=0.0)

        # 可选姿态项：鼓励脚底 roll/pitch 接近水平，避免歪脚落地。
        foot_quat = self.rigid_state[:, self.feet_indices, 3:7].reshape(-1, 4)
        foot_rp = get_euler_xyz_tensor(foot_quat).reshape(self.num_envs, -1, 3)[:, :, 0:2]
        foot_rp_error = torch.norm(foot_rp, dim=2)

        penalty = (
            torch.square(rp_ang_vel_excess)
            + 0.2 * torch.square(foot_rp_error)
        ) * pre_landing_mask
        return torch.sum(penalty, dim=1)

    def _reward_foot_slip2(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """

        feet_force = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2)
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)

        error_buf =  torch.logical_and(feet_force[:, 0] > 100, feet_z[:, 0] > 0.1)
        error_buf |=  torch.logical_and(feet_force[:, 1] > 100, feet_z[:, 1] > 0.1)

        error_buf |=  torch.logical_and(feet_force[:, 0] > 20, foot_speed_norm[:, 0] > 0.5)
        error_buf |=  torch.logical_and(feet_force[:, 1] > 20, foot_speed_norm[:, 1] > 0.5)

        reward = torch.where(error_buf, 1., 0.)
        return reward

    def _reward_foot_pitch(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """

        swing_mask = 1 - self._get_gait_phase()
        foot_rot = self.rigid_state[:, self.feet_indices, 3:7]
        foot_rot_0 = foot_rot[:, 0]
        foot_rot_1 = foot_rot[:, 1]
        foot_rot_0 = get_euler_xyz_tensor(foot_rot_0)[:, 1]
        foot_rot_1 = get_euler_xyz_tensor(foot_rot_1)[:, 1]

        error = torch.square(foot_rot_0)*swing_mask[:, 0] + torch.square(foot_rot_1)*swing_mask[:, 1]
        reward = torch.exp(error * -100.)
        return reward

    def _reward_feet_air_time(self):
        """
        奖励足端腾空时间：仅在首次触地时，根据 (实际腾空时间 - 目标) * first_contact 给奖，
        air_time 限制为 <=0（不足目标为负，超过目标截断为 0）；且仅在参考根速度有水平分量时给奖。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        tgt_air_time = self.cfg.rewards.feet_air_time_target
        air_time = (self.feet_air_time - tgt_air_time) * first_contact
        air_time = air_time.clamp(max=0.)
        self.feet_air_time *= ~self.contact_filt
        rew_airtime = air_time.sum(dim=1)
        # 仅在有参考水平速度时给奖（避免站立时被误奖）
        ref_root_vel_xy = self.ref_root_linvel[:, 0:2]
        rew_airtime *= (torch.norm(ref_root_vel_xy, dim=1) > 0.05).float()
        return rew_airtime
    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - self.config.rewards.desired_feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
    #     # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     # print("Rew air time: ", rew_airTime)
    #     return rew_airTime

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_knee_limit(self):
        """
        Calculates a reward based on the number of knee joint.
        """

        knee_joint_left = self.dof_pos[:, 3]
        knee_joint_right = self.dof_pos[:, 9]


        reward_left = torch.where(knee_joint_left < 0.15, 1., 0.)
        reward_right = torch.where(knee_joint_right < 0.15, 1., 0.)
        return (reward_left + reward_right)/2.

    # def _reward_orientation(self):
    #     """
    #     Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
    #     from the desired base orientation using the base euler angles and the projected gravity vector.
    #     """
    #     quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
    #     orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
    #     return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        from_air_to_ground = torch.logical_and(contact_filt, ~self.last_contacts_filt)
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt
        foot_rot = self.rigid_state[:, self.feet_indices, 3:7]
        foot_rot_0 = foot_rot[:, 0]
        foot_rot_1 = foot_rot[:, 1]
        foot_rot_0 = get_euler_xyz_tensor(foot_rot_0)[:, 0]
        foot_rot_1 = get_euler_xyz_tensor(foot_rot_1)[:, 0]

        reward = torch.square(foot_rot_0)*from_air_to_ground[:, 0] + torch.square(foot_rot_1)*from_air_to_ground[:, 1]

        return reward

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
   
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 5000), dim=1)

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_vel = self.rigid_state[:, self.feet_indices, 7:10]
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target

        heading_rot = torch_utils.calc_heading_quat_inv(self.base_quat)

        knee_rot = self.rigid_state[:, self.ankle_indices, 3:7]
        knee_rot_0 = quat_mul(heading_rot, knee_rot[:, 0])
        knee_rot_1 = quat_mul(heading_rot, knee_rot[:, 1])
        knee_rot_0 = get_euler_xyz_tensor(knee_rot_0)
        knee_rot_1 = get_euler_xyz_tensor(knee_rot_1)

        left_yaw_roll = knee_rot_0[:, [0,2]]
        right_yaw_roll = knee_rot_1[:, [0,2]]
        left_yaw_roll[:, :] *= 0.
        right_yaw_roll[:, :] *= 0.

        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.3, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        root_linvel_xyz = self.base_lin_vel - self.ref_root_linvel

        lin_vel_error = torch.norm(
            self.commands[:, :2] - root_linvel_xyz[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """

        root_linvel_xyz = self.base_lin_vel - self.ref_root_linvel

        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - root_linvel_xyz[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_stand(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        diff = (self.dof_pos - self.default_dof_pos)*0.5
        diff[:, [3, 9]] *= 4.

        r = 0.2 * torch.norm(diff, dim=1).clamp(0, 5.)

        reward = torch.where(torch.abs(self.commands[:, 0]) <= 0.1, r, r*0.)
        return reward

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        root_linvel_xyz = self.base_lin_vel - self.ref_root_linvel

        absolute_speed = torch.abs(root_linvel_xyz[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            root_linvel_xyz[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(root_linvel_xyz[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    # def _reward_torques(self):
    #     """
    #     Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
    #     the necessary force exerted by the motors.
    #     """
    #     # self.torques[:, [3, 13]] *= 2
    #     torques_1 = self.torques.clone()[:, [4, 10]]
    #     torques_1 = (torch.abs(torques_1) - 20.).clip(0, 100)
    #     torques_1 *= 50.

    #     torques_2 = self.torques.clone()[:, [5, 11]]
    #     torques_2 = (torch.abs(torques_2) - 15.).clip(0, 100)
    #     torques_2 *= 50.

    #     torques_3 = self.torques.clone()[:, [3, 9]]
    #     torques_3 = (torch.abs(torques_3) - 90.).clip(0, 100)
    #     torques_3 *= 20.
        
    #     return torch.sum(torch.square(self.torques[:, self.num_control]), dim=1) + torch.sum(torch.square(torques_1), dim=1) + torch.sum(torch.square(torques_2) , dim=1) + torch.sum(torch.square(torques_3) , dim=1)

    def _reward_torques(self):
        # Penalize torques
        
        return torch.sum(torch.square(self.torques[:, self.num_control]), dim=1)
    
    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel[:, self.num_control]), dim=1)

    def _reward_energy(self):
        """
        Penalizes high energy
        """

        energy_ = torch.sum(torch.abs(self.dof_vel*self.torques), dim=1)
        w = torch.where(self.last_contacts[:,0], 5., 1.)
        return energy_ * w

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, self.num_control] / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, self.num_control] - self.actions[:, self.num_control]), dim=1)

    def _reward_waist_action(self):
        # Penalize the residual action on the waist joint to reduce overuse of waist compensation.
        waist_idx = 12
        if waist_idx not in self.num_control:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.square(self.actions[:, waist_idx])
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    # def _reward_action_smoothness(self):
    #     """
    #     Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
    #     This is important for achieving fluid motion and reducing mechanical stress.
    #     """
    #     term_1 = torch.sum(torch.square(
    #         self.last_actions[:, self.num_control] - self.actions[:, self.num_control]), dim=1)
    #     term_2 = torch.sum(torch.square(
    #         self.actions[:, self.num_control] + self.last_last_actions[:, self.num_control] - 2 * self.last_actions[:, self.num_control]), dim=1)
    #     term_3 = 0.05 * torch.sum(torch.abs(self.actions[:, self.num_control]), dim=1)
    #     return term_1 + term_2 + term_3
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.actions[:, self.num_control] + self.last_last_actions[:, self.num_control] - 2 * self.last_actions[:, self.num_control]), dim=1)
        return term_1

    def _reward_torques_smoothness(self):
        """
        Encourages smoothness in the robot's torques
        """
        r = torch.sum(torch.square(
            self.last_torques[:, self.num_control] - self.torques[:, self.num_control]), dim=1)

        return r

# ================================================ imition Rewards ============================================== #
    def _reward_imition_root_height(self):
        ref_root_height = self.ref_waist_pos[:, 0, 2:3]
        cur_waist_pos, _ = self._get_current_anchor_pose_local()

        diff = (cur_waist_pos[:, 2:3] - ref_root_height)
        r = torch.exp(-torch.square(torch.norm(diff, dim=1)) / self.cfg.rewards.sigma.root_height**2)

        return r


    def _reward_imition_torso_orientation(self):
        euler_xy = self.ref_euler_xyz[:, 0:2]

        diff = torch.abs(euler_xy - self.base_euler_xyz[:, 0:2])
        diff[diff > np.pi] -= 2 * np.pi
        
        r = torch.exp(-100. * torch.square(torch.norm(diff, dim=1)))
        return r

    def _reward_imition_torso_yaw(self):
        euler_z = self.ref_euler_xyz[:, 2:3]

        diff = torch.abs(euler_z - self.base_euler_xyz[:, 2:3])
        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi
        
        r = torch.exp(-30. * torch.square(torch.norm(diff, dim=1)))
        return r

    def _reward_imition_linear_velocity_x(self):
        root_linvel_xy = self.ref_root_linvel[:, 0:1]

        diff = (self.base_lin_vel[:, 0:1] - root_linvel_xy)
        r = torch.exp(-80. * torch.square(torch.norm(diff, dim=1)))
        # r = torch.square(torch.norm(diff, dim=1))
        return r

    def _reward_imition_linear_velocity_y(self):
        root_linvel_xy = self.ref_root_linvel[:, 1:2]

        diff = root_linvel_xy - self.base_lin_vel[:, 1:2]
        r = torch.exp(-8. * torch.square(torch.norm(diff, dim=1)))
        return r

    def _reward_imition_linear_velocity_z(self):
        root_linvel_z = self.ref_root_linvel[:, 2:]

        diff = root_linvel_z - self.base_lin_vel[:, 2:]
        r = torch.exp(-8. * torch.square(torch.norm(diff, dim=1)))
        return r
    
    def _reward_imition_angular_velocity_xy(self):
        root_angvel_xy = self.ref_root_angvel[:, 0:2]

        diff = root_angvel_xy - self.base_ang_vel[:, 0:2]
        r = torch.exp(-50. * torch.square(torch.norm(diff, dim=1)))
        return r

    def _reward_imition_angular_velocity_z(self):
        root_angvel_z = self.ref_root_angvel[:, 2:]

        diff = root_angvel_z - self.base_ang_vel[:, 2:]
        r = torch.exp(-2. * torch.square(torch.norm(diff, dim=1)))
        return r

    def _reward_imition_joint_pos_leg(self):
        pos_target = self.ref_dof_pos[:, self.num_control]
        joint_pos = self.dof_pos.clone()[:, self.num_control]

        diff = (joint_pos - pos_target)
        # diff[:, [0, 6, 4, 10]] *= 2.
        # diff[:, [3, 9]] *= 5.
        # diff[:, [4, 10]] *= 2.
        # diff[:, [0, 3, 6, 9]] *= 5.

        r = torch.square(torch.norm(diff, dim=1))
        return -r

    def _reward_imition_joint_pos_arm(self):
        pos_target = self.ref_dof_pos[:, self.num_control]
        joint_pos = self.dof_pos.clone()[:, self.num_control]

        diff = (joint_pos - pos_target)[:, [6, 7, 8, 15, 16, 17]]
        # diff *= 0.5
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_imition_mirr_joint_pos(self):
        mirr_ = np.array([-1., -1., -1., 1., 1., -1., 1., -1., -1., 1., 1., -1., -1., -1., -1., 1., 1., -1., 1., -1., -1., 1., 1., -1., 1, 1, 1, 1, 1])
        # mirr_ = np.array([-1., -1., -1., 1., 1., -1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., 1., -1., -1., 1., 1., -1.])
        mirr_ = torch.from_numpy(mirr_).requires_grad_(False).to(self.device)
        left_right_pos_diff = self.dof_pos[:, :6] - (self.dof_pos*mirr_)[:, 6:12]
        left_right_pos_diff = torch.norm(left_right_pos_diff[:, 0:6], dim=1)

        r = torch.exp(-10. * torch.square(left_right_pos_diff))
        return r

    # def _reward_imition_joint_vel(self):
    #     dof_vel_target = self.dof_vel_buffer[self.ref_idx, self.phase_idx][:, self.num_control].clone()
    #     dof_vel_target = dof_vel_target.clamp(-10., 10.)

    #     diff = dof_vel_target - self.dof_vel[:, self.num_control]
    #     # diff[:, 4:6] *= 2.
    #     # diff[:, 10:12] *= 2.
    #     r = torch.square(torch.norm(diff, dim=1))

    #     return -r

    def _reward_imition_foot_height(self):
        # swing_mask = 1 - self._get_gait_phase()
        # more_clearence = 0.05 * torch.ones(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        # no_clearence = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        # more_clearence = torch.where(swing_mask.bool(), more_clearence, no_clearence)
        foot_height = self.ref_foot_height
        feet_z = self.rigid_state[:, self.feet_indices, 2]

        diff = (feet_z - foot_height)
        r = torch.exp(-torch.square(torch.norm(diff, dim=1)) / self.cfg.rewards.sigma.foot_height**2)

        return r
        # r = torch.square(torch.norm(diff, dim=1))
        # return -r


    
    def _reward_imition_keybody_vel(self):
  
        feet_vel_ref = self.ref_feet_vel
        key_body_feet_vel = self.rigid_state[:, self.feet_indices, 7:10].clone()
        diff_sq = torch.square(key_body_feet_vel - feet_vel_ref).sum(dim=-1).mean(dim=1)
        sigma_vel = 0.5 
        r = torch.exp(-sigma_vel * diff_sq)

        return r
    
    def _reward_imition_keybody_orientation(self):
        ref_quat = self.ref_feet_quat
        cur_quat = self.rigid_state[:, self.feet_indices, 3:7] 
        inner_product = torch.sum(cur_quat * ref_quat, dim=-1)
        quat_diff_sq = 1.0 - torch.square(inner_product)
        total_rot_error = torch.mean(quat_diff_sq, dim=1)
        sigma_quat = 40.0
        r = torch.exp(-sigma_quat * total_rot_error)

        return r
    
    def _reward_imition_keybody_euler(self):
        feet_euler_target = get_euler_xyz_tensor(self.ref_feet_quat.reshape(-1, 4)).reshape(self.num_envs, -1, 3)

        # feet
        left_feet_quat = self.rigid_state[:, 0, 3:7].clone()
        left_feet_euler = get_euler_xyz_tensor(left_feet_quat)
        right_feet_quat = self.rigid_state[:, 1, 3:7].clone()
        right_feet_euler = get_euler_xyz_tensor(right_feet_quat)
        feet_euler = torch.stack((left_feet_euler, right_feet_euler), dim=1)
        diff = feet_euler_target - feet_euler
        diff[:, :, 2] *= 0
        r = torch.exp(-300. * torch.square(diff).mean(dim=-1).mean(dim=-1))

        return r

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        control_idx = self.num_control
        dof_pos = self.dof_pos[:, control_idx]
        dof_limits = self.dof_pos_limits[control_idx]
        out_of_limits = -(dof_pos - dof_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (dof_pos - dof_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     dof_vel_limit = (torch.abs(self.dof_vel[:, 4]) - 8.).clip(min=0., max=1.) + (torch.abs(self.dof_vel[:, 10]) - 8.).clip(min=0., max=1.)
    #     return dof_vel_limit
    def _reward_dof_vel_limits(self):
        
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits).clip(min=0., max=1.),dim=1)
    
    def _reward_torque_limits(self):
        soft_limit_val = self.torque_limits * 0.9
        torques_to_check = torch.abs(self.torques[:, self.num_control])
        
        relevant_soft_limits = soft_limit_val[self.num_control]
        over_limit = torques_to_check - relevant_soft_limits
        violation = torch.clamp(over_limit, min=0.)
        reward = torch.mean(violation / (self.torque_limits[self.num_control] * 0.1), dim=1)
        
        return torch.clamp(reward, min=0., max=1.)
    
    def _reward_torque_penalty(self):
        # 只有受控关节参与惩罚
        tau_norm = self.torques_raw[:, self.num_control] / self.torque_limits[self.num_control]
        return torch.mean(tau_norm**2, dim=1)


    def _reward_imition_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, 0.)


        # self.phase_idx = self.episode_phase_buf % self.demo_length
        # self.last_contacts[:,0] = torch.logical_and(self.phase_idx > 180,  torch.logical_or(contact[:, 0], contact[:, 1]))
        # self.last_contacts[:,1] = torch.logical_and(self.phase_idx > 180,  torch.logical_or(contact[:, 0], contact[:, 1]))

        return torch.sum(reward, dim=1)

    def _reward_imition_survival(self):
        return 1.

    def _reward_imition_leg_joint_pos_error_exp(self):
        pos_target = self.ref_dof_pos[:, self.num_control]
        joint_pos = self.dof_pos.clone()[:, self.num_control]

        diff = joint_pos - pos_target
        weights = torch.ones_like(diff)
        weights[:, [0, 3, 6, 9]] = 3.0 
        diff_sq = torch.sum(torch.square(diff * weights), dim=1)
        sigma = 15.0 
        r = torch.exp(-sigma * diff_sq)
        
        return r
    
    def _reward_imition_leg_joint_vel_error_exp(self):

        dof_vel_target = self.ref_dof_vel[:, self.num_control].clone()
        dof_vel_target = dof_vel_target.clamp(-10., 10.)
        diff = dof_vel_target - self.dof_vel[:, self.num_control]

        diff_sq = torch.sum(torch.square(diff), dim=1)
        sigma_vel = 0.1
        r = torch.exp(-sigma_vel * diff_sq)

        return r
    
    def _reward_imition_keybody_pos(self):
        ref_body_pos, _, _, _ = self._tracking_ref_tensors()
        cur_body_pos, target_pos = self._get_aligned_body_pos_targets(self.all_tracking_indices, ref_body_pos)
        self.key_body_diff = cur_body_pos - target_pos
        dist_sq = torch.sum(torch.square(self.key_body_diff), dim=(1, 2))
        sigma_key = 50.0 
        r = torch.exp(-sigma_key * dist_sq)

        return r
    
   

    def _reward_root_xy_stay(self):
        """惩罚根位置在 X 方向上的漂移（仅 X，不含 Y），使机器人尽量原地舞蹈。"""
        ref_root_x = self.ref_waist_pos[:, 0, 0:1]
        cur_root_x = self.root_states[:, 0:1] - self.env_origins[:, 0:1]
        diff_x_sq = torch.square(cur_root_x - ref_root_x).squeeze(-1)
        sigma_x = 0.3
        return torch.exp(-diff_x_sq / sigma_x**2)

    def _reward_root_rot_stay(self):
        """惩罚机身旋转偏离参考姿态，使机器人尽量保持与参考一致的朝向。"""
        ref_root_quat = self.ref_waist_quat[:, 0, :]
        cur_root_quat = self.root_states[:, 3:7]
        rot_error_rad = quat_error_magnitude(cur_root_quat, ref_root_quat)
        sigma_rot = 0.15  # 弧度，约 14°，旋转误差越小奖励越高
        return torch.exp(-torch.square(rot_error_rad) / sigma_rot**2)

    def _reward_torso_yaw_stay(self):
        """惩罚机身锚点偏航角偏离参考，使朝向与参考一致。仅考虑 yaw，不含 roll/pitch。"""
        ref_root_quat = self.ref_waist_quat[:, 0, :]
        cur_root_quat = self.root_states[:, 3:7]
        cur_yaw = get_euler_xyz_tensor(cur_root_quat)[:, 2]
        ref_yaw = get_euler_xyz_tensor(ref_root_quat)[:, 2]
        yaw_diff = cur_yaw - ref_yaw
        yaw_diff = torch.where(yaw_diff > np.pi, yaw_diff - 2 * np.pi, yaw_diff)
        yaw_diff = torch.where(yaw_diff < -np.pi, yaw_diff + 2 * np.pi, yaw_diff)
        sigma_yaw = 0.18  # 弧度，约 17°
        return torch.exp(-torch.square(yaw_diff) / sigma_yaw**2)

        
    def _reward_imition_feet_pos(self):
        ref_feet_pos = self.ref_feet_pos
        cur_feet_pos, target_pos = self._get_aligned_body_pos_targets(self.feet_indices, ref_feet_pos)
        dist_sq = torch.sum(torch.square(cur_feet_pos - target_pos), dim=-1).mean(dim=1)
        return torch.exp(-dist_sq / self.cfg.rewards.sigma.feet_pos**2)
    
    
    def _reward_imition_knee_pos(self):
        ref_knee_pos = self.ref_knee_pos
        cur_knee_pos, target_pos = self._get_aligned_body_pos_targets(self.knee_indices, ref_knee_pos)
        dist_sq = torch.sum(torch.square(cur_knee_pos - target_pos), dim=-1).mean(dim=1)
        sigma_key = 80.0
        return torch.exp(-sigma_key * dist_sq)
        
    def _reward_imition_hip_pos(self):
        ref_hip_pos = self.ref_hip_pos
        cur_hip_pos, target_pos = self._get_aligned_body_pos_targets(self.hip_indices, ref_hip_pos)
        dist_sq = torch.sum(torch.square(cur_hip_pos - target_pos), dim=-1).mean(dim=1)
        sigma_key = 80.0
        return torch.exp(-sigma_key * dist_sq)
    
    def _reward_imition_pelvis_pos(self):
        ref_pelvis_pos = self.ref_pelvis_pos
        cur_pelvis_pos, target_pos = self._get_aligned_body_pos_targets(self.base_indices, ref_pelvis_pos)
        dist_sq = torch.sum(torch.square(cur_pelvis_pos - target_pos), dim=-1).mean(dim=1)
        sigma_key = 90.0
        return torch.exp(-sigma_key * dist_sq)

    def _reward_imition_pelvic_yaw_pos(self):
        ref_pelvic_yaw_pos = self.ref_pelvic_yaw_pos
        cur_pelvic_yaw_pos, target_pos = self._get_aligned_body_pos_targets(self.pelvic_yaw_indices, ref_pelvic_yaw_pos)
        dist_sq = torch.sum(torch.square(cur_pelvic_yaw_pos - target_pos), dim=-1).mean(dim=1)
        sigma_key = 90.0
        return torch.exp(-sigma_key * dist_sq)

    def _reward_imition_waist_pos(self):
        ref_waist_pos = self.ref_waist_pos
        cur_waist_pos, target_pos = self._get_aligned_body_pos_targets(self.waist_indices, ref_waist_pos)
        dist_sq = torch.sum(torch.square(cur_waist_pos - target_pos), dim=-1).mean(dim=1)
        sigma_key = 90.0
        return torch.exp(-sigma_key * dist_sq)
        
    def _reward_imition_feet_rot(self):
        feet_quat_ref = self.ref_feet_quat
        current_feet_quat, target_quat = self._get_aligned_body_quat_targets(self.feet_indices, feet_quat_ref)
        rot_error_rad = quat_error_magnitude(current_feet_quat, target_quat)
        return torch.exp(-torch.square(rot_error_rad).mean(dim=-1) / self.cfg.rewards.sigma.feet_rot**2)

    def _reward_imition_knee_rot(self):
        knee_quat_ref = self.ref_knee_quat
        current_knee_quat, target_quat = self._get_aligned_body_quat_targets(self.knee_indices, knee_quat_ref)
        rot_error_rad = quat_error_magnitude(current_knee_quat, target_quat)
        sigma_keybody_rot = 20.0
        return torch.exp(-sigma_keybody_rot * torch.square(rot_error_rad).mean(dim=-1))
    
    def _reward_imition_hip_rot(self):
        hip_quat_ref = self.ref_hip_quat
        current_hip_quat, target_quat = self._get_aligned_body_quat_targets(self.hip_indices, hip_quat_ref)
        rot_error_rad = quat_error_magnitude(current_hip_quat, target_quat)
        sigma_keybody_rot = 20.0
        return torch.exp(-sigma_keybody_rot * torch.square(rot_error_rad).mean(dim=-1))
    
    def _reward_imition_pelvis_rot(self):
        pelvis_quat_ref = self.ref_pelvis_quat
        current_pelvis_quat, target_quat = self._get_aligned_body_quat_targets(self.base_indices, pelvis_quat_ref)
        rot_error_rad = quat_error_magnitude(current_pelvis_quat, target_quat)
        sigma_keybody_rot = 20.0
        return torch.exp(-sigma_keybody_rot * torch.square(rot_error_rad).mean(dim=-1))

    def _reward_imition_pelvic_yaw_rot(self):
        pelvic_yaw_quat_ref = self.ref_pelvic_yaw_quat
        current_pelvic_yaw_quat, target_quat = self._get_aligned_body_quat_targets(self.pelvic_yaw_indices, pelvic_yaw_quat_ref)
        rot_error_rad = quat_error_magnitude(current_pelvic_yaw_quat, target_quat)
        sigma_keybody_rot = 20.0
        return torch.exp(-sigma_keybody_rot * torch.square(rot_error_rad).mean(dim=-1))
    
    def _reward_imition_waist_rot(self):
        waist_quat_ref = self.ref_waist_quat
        current_waist_quat, target_quat = self._get_aligned_body_quat_targets(self.waist_indices, waist_quat_ref)
        rot_error_rad = quat_error_magnitude(current_waist_quat, target_quat)
        sigma_keybody_rot = 20.0
        return torch.exp(-sigma_keybody_rot * torch.square(rot_error_rad).mean(dim=-1))

    def _reward_imition_keybody_lin_vel(self):
        """关键点线速度跟踪奖励"""
        _, _, ref_key_vel, _ = self._tracking_ref_tensors()
        cur_key_vel, target_key_vel = self._get_aligned_body_vector_targets(self.all_tracking_indices, ref_key_vel, slice(7, 10))
        lin_vel_error = torch.sum(torch.square(cur_key_vel - target_key_vel), dim=-1).mean(dim=-1)

        sigma_lin_vel = 1.0 
        return torch.exp(-lin_vel_error / sigma_lin_vel)
    
    def _reward_imition_keybody_ang_vel(self):
        """关键点角速度跟踪奖励 """
        _, _, _, ref_key_ang_vel = self._tracking_ref_tensors()
        cur_key_ang_vel, target_key_ang_vel = self._get_aligned_body_vector_targets(self.all_tracking_indices, ref_key_ang_vel, slice(10, 13))
        ang_vel_error = torch.sum(torch.square(cur_key_ang_vel - target_key_ang_vel), dim=-1).mean(dim=-1)
        sigma_ang_vel = 0.5 
        return torch.exp(-ang_vel_error / sigma_ang_vel)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non-flat base orientation
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
        
    def _reward_imition_root_rot(self):
        ref_root_quat = self.ref_waist_quat[:, 0, :]
        _, cur_root_quat = self._get_current_anchor_pose()
        rot_error_rad = quat_error_magnitude(cur_root_quat, ref_root_quat)
        
        # sigma_root_rot = 0.4
        return torch.exp(-torch.square(rot_error_rad) / self.cfg.rewards.sigma.root_rot**2)

    def _reward_imition_base_lin_vel(self):
        ref_lin_vel = self.ref_root_linvel
        cur_lin_vel = self.base_lin_vel
        diff_sq = torch.sum(torch.square(cur_lin_vel - ref_lin_vel), dim=-1)
        sigma_lin_vel = 5.0 
        return torch.exp(-sigma_lin_vel * diff_sq)
    
    def _reward_imition_base_ang_vel(self):
        ref_ang_vel = self.ref_waist_ang_vel[:, 0, :]
        waist_idx = int(self.waist_indices[0].item())
        cur_ang_vel = self.rigid_state[:, waist_idx, 10:13]
        diff_sq = torch.sum(torch.square(cur_ang_vel - ref_ang_vel), dim=-1)
        return torch.exp(-diff_sq / self.cfg.rewards.sigma.root_ang_vel**2)
    
    def _reward_imition_root_pos(self):
        ref_root_pos = self.ref_waist_pos[:, 0, :]
        cur_root_pos, _ = self._get_current_anchor_pose_local()
        diff_sq = torch.sum(torch.square(cur_root_pos - ref_root_pos), dim=-1)
        return torch.exp(-diff_sq / self.cfg.rewards.sigma.root_pos**2)

    def _reward_imitation_root_vel(self):

        ref_root_vel = self.ref_waist_vel[:, 0, :]
        waist_idx = int(self.waist_indices[0].item())
        cur_root_vel = self.rigid_state[:, waist_idx, 7:10]

        diff_sq = torch.sum(torch.square(cur_root_vel - ref_root_vel), dim=-1)
        return torch.exp(-diff_sq / self.cfg.rewards.sigma.root_vel**2)
    
    def _reward_imitation_whole_body_pos(self):
        # A. 准备参考数据 (Reference Data) - 必须和索引顺序一致！
        ref_body_pos, _, _, _ = self._tracking_ref_tensors()
        cur_body_pos, target_pos = self._get_aligned_body_pos_targets(self.all_tracking_indices, ref_body_pos)
        dist_sq = torch.sum(torch.square(cur_body_pos - target_pos), dim=-1)
        mean_dist_sq = dist_sq.mean(dim=1)
        return torch.exp(-mean_dist_sq / self.cfg.rewards.sigma.whole_body_pos**2)
    
    def _reward_imitation_whole_body_rot(self):
        # A. 准备参考数据 (Reference Data)
        _, ref_body_quat, _, _ = self._tracking_ref_tensors()
        cur_body_quat, target_quat = self._get_aligned_body_quat_targets(self.all_tracking_indices, ref_body_quat)
        rot_error_rad = torch_utils.quat_error_magnitude(cur_body_quat, target_quat)
        return torch.exp(-torch.square(rot_error_rad).mean(dim=-1) / self.cfg.rewards.sigma.whole_body_rot**2)
    
    def _reward_imitation_whole_body_lin_vel(self):
        """全身（含Base）线速度跟踪奖励 - 带Yaw重定向"""
        _, _, ref_body_vel, _ = self._tracking_ref_tensors()
        cur_body_vel, target_vel = self._get_aligned_body_vector_targets(self.all_tracking_indices, ref_body_vel, slice(7, 10))
        lin_vel_error = torch.sum(torch.square(cur_body_vel - target_vel), dim=-1).mean(dim=-1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.sigma.whole_body_lin_vel**2)
    
    def _reward_imitation_whole_body_ang_vel(self):
        _, _, _, ref_body_ang_vel = self._tracking_ref_tensors()
        cur_body_ang_vel, target_ang_vel = self._get_aligned_body_vector_targets(self.all_tracking_indices, ref_body_ang_vel, slice(10, 13))
        ang_vel_error = torch.sum(torch.square(cur_body_ang_vel - target_ang_vel), dim=-1).mean(dim=-1)
        return torch.exp(-ang_vel_error / self.cfg.rewards.sigma.whole_body_ang_vel**2)
    
    def _reward_ankle_regularization(self):
        """
        惩罚脚踝关节偏离默认位置。
        脚踝是每条腿的第 5 和 第 6 个关节。
        索引对应：
        Left Leg: 0-5 -> Ankle: 4, 5
        Right Leg: 6-11 -> Ankle: 10, 11
        """
        ankle_indices = [ 5,  11]
        
        current_ankle_pos = self.dof_pos[:, ankle_indices]
        target_ankle_pos = self.default_dof_pos[:, ankle_indices]
        diff = current_ankle_pos - target_ankle_pos
        return torch.sum(torch.square(diff), dim=1)

    def _reward_ankle_action_rate(self):
        """
        惩罚脚踝关节动作变化过快（防止抖动）。
        """
        ankle_indices = [ 4, 5, 10, 11]
        diff = self.actions[:, ankle_indices] - self.last_actions[:, ankle_indices]
        
        return torch.sum(torch.square(diff), dim=1)
    
    def _reward_ankle_dof_acc(self):
        # ankle_dof_idx = [4, 5, 10, 11]
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        # ankle_dof_idx = [5, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)

    def _reward_ankle_roll_dof_vel(self):
        """只惩罚左右 ankle roll 速度，压制脚底内外侧支撑引起的 roll 高频抖动。"""
        ankle_roll_dof_idx = [5, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_roll_dof_idx]), dim=1)

    def _reward_ankle_torque_limit(self):
        """惩罚踝关节力矩超限，避免实机踝关节过载。踝关节索引：左 4,5 右 10,11。"""
        ankle_dof_idx = [4, 5, 10, 11]
        ankle_tau_limits = self.torque_limits[ankle_dof_idx]
        ankle_torques = torch.abs(self.torques[:, ankle_dof_idx])
        soft_ratio = 0.9
        soft_limits = ankle_tau_limits * soft_ratio
        over_limit = (ankle_torques - soft_limits).clamp(min=0.)
        violation = torch.sum(over_limit / (ankle_tau_limits * 0.15 + 1e-6), dim=1)
        return torch.clamp(violation, min=0., max=1.)
    
    # def _reward_imition_joint_pos(self):
    #     pos_target = self.dof_pos_buffer[self.ref_idx, self.phase_idx][:, self.num_control]
    #     joint_pos = self.dof_pos.clone()[:, self.num_control]

    #     diff = (joint_pos - pos_target)
    #     # diff[:, [0, 6, 4, 10]] *= 2.
    #     # diff[:, [3, 9]] *= 5.
    #     # diff[:, [4, 10]] *= 2.
    #     diff[:, [0, 3, 6, 9]] *= 5.
    #     sigma_dof_pos = 0.5
    #     r = torch.square(torch.norm(diff, dim=1))
    #     return torch.exp(-r / sigma_dof_pos**2)
    

    def _reward_swing_feet_height(self):
        """
        仅在摆动相（swing）对足端高度跟踪给奖励：参考轨迹里该足离地（ref_z > 0.08）时，
        鼓励当前足端高度与参考一致；支撑足不参与。用高斯 exp(-误差²/sigma) 形式，对两足取平均。
        """
        # 当前足端 z，高度 (num_envs, num_feet)
        cur_z = self.rigid_state[:, self.feet_indices, 2]

        # 参考足端 z，高度 (num_envs, num_feet)
        current_ref_pose = self.ref_feet_pos
        ref_z = current_ref_pose[:, :, 2]

        # 判断 swing：参考足高度超过阈值
        swing_mask = ref_z > 0.07
        # swing_mask: (num_envs, num_feet), bool

        # 高度误差
        height_error = cur_z - ref_z

        sigma = 0.06
        per_foot_reward = torch.exp(-(height_error ** 2) / sigma)
        per_foot_reward = per_foot_reward * swing_mask.float()

        # 只对摆动足取平均，梯度不被支撑足稀释
        n_swing = swing_mask.sum(dim=1).clamp(min=1e-6)
        reward = (per_foot_reward.sum(dim=1) / n_swing)
        return reward

    def _reward_imition_joint_vel(self):
        # 获取目标关节速度
        dof_vel_target = self.ref_dof_vel[:, self.num_control].clone()
        dof_vel_target = dof_vel_target.clamp(-10., 10.)

        diff = dof_vel_target - self.dof_vel[:, self.num_control]
        r = torch.sum(self.dof_err_w * torch.square(diff), dim=1)
        # 指数形式奖励：使用较小的 sigma 保证灵敏度，建议值在 1.0 - 4.0 之间
        sigma_dof_vel = 2.0 
        return torch.exp(-r / sigma_dof_vel**2)

    def _reward_imition_joint_pos(self):
        # 获取目标关节位置
        pos_target = self.ref_dof_pos[:, self.num_control]
        joint_pos = self.dof_pos.clone()[:, self.num_control]
        diff = joint_pos - pos_target
        r = torch.sum(self.dof_err_w * torch.square(diff), dim=1)
        sigma_dof_pos = 0.5
        return torch.exp(-r / sigma_dof_pos**2)

    def _reward_penalty_stumble(self):
        # 惩罚足端撞到竖直面（如墙、台阶侧面）：水平接触力相对法向力过大时视为“绊到”
        # 条件：水平力范数 > 5*|法向力| 时记为一次 stumble，任一足满足即惩罚
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_teleop_contact_mask(self):
        # 当前足端接触：用较高阈值(20N)避免脚尖拖地被判成“触地”，真实支撑通常远大于 20N
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        cur_contact_mask = (contact_force_z > 20.0).float()
        # 参考足端接触：ref_foot_contact（1=抬脚，0=不抬脚）已转为接触掩码(1=触地)存入 feet_contact_buffer
        ref_contact_mask = self.ref_feet_contact.float()
        # contact mismatch
        error_contact_mask = torch.abs(cur_contact_mask - ref_contact_mask)
        rew = 1.0 - error_contact_mask.mean(dim=-1)
        # 拖地惩罚：参考应抬脚时仍存在明显接触力 → 脚尖拖地，额外惩罚
        ref_lift = (1.0 - ref_contact_mask).float()
        drag_force = (contact_force_z - 20.0).clamp(min=0.0) * ref_lift
        rew = rew - 0.01 * drag_force.mean(dim=-1)  # 轻微惩罚脚尖拖地，避免拖地得利
        return rew

    def _reward_ref_lift_foot_clearance(self):
        """参考该脚应抬腿时，仅鼓励抬到阈值以上（达标=1，未达标=0）。"""
        ref_z = self.ref_foot_height
        # 应抬脚：由参考接触掩码取反得到；同时保留参考脚高作为补充判断
        ref_lift_from_contact = (1.0 - self.ref_feet_contact).float()
        ref_lift_from_height = (ref_z > 0.08).float()
        ref_lift_mask = torch.maximum(ref_lift_from_contact, ref_lift_from_height)
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        # 足端世界系高度超过阈值即记为达标
        clearance_threshold = 0.11
        rew_per_foot = (feet_z > clearance_threshold).float()
        n_lift = ref_lift_mask.sum(dim=1).clamp(min=1e-6)
        rew = (rew_per_foot * ref_lift_mask).sum(dim=1) / n_lift
        return rew
    
    # 新增奖励：鼓励 pitch 略微前倾（比如 0.02 rad ≈ 1°）
    def _reward_forward_lean(self):
        pitch = self.base_euler_xyz[:, 1]  # pitch，正值=前倾
        return torch.clamp(pitch, 0.0, 0.05)

    def _reward_com_over_support_foot(self):
        """
        运动质心奖励：鼓励质心投影落在支撑脚上方，取值 [0, 1]，乘以权重 2.0 后最终 [0, 2.0]。
        步骤：1) c_xy 质心 XY；2) f_min_xy 支撑脚 XY；3) 欧氏距离；4) exp(-d^2/sigma^2) 得基础奖励；
        5) 乘以 I_unbiased（失衡则置 0）。
        """
        c_xy = self.root_states[:, 0:2]
        feet_xy = self.rigid_state[:, self.feet_indices, 0:2]
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_threshold = 20.0
        contact_mask = (contact_force_z > contact_threshold).float()
        n_contact = contact_mask.sum(dim=1).clamp(min=1e-6)
        f_min_xy = (feet_xy * contact_mask.unsqueeze(-1)).sum(dim=1) / n_contact.unsqueeze(-1)
        has_contact = (contact_mask.sum(dim=1) > 0.5).float()
        dist_sq = torch.sum(torch.square(c_xy - f_min_xy), dim=1)
        sigma = getattr(self.cfg.rewards.sigma, 'com_over_support_foot', 0.08)
        base_rew = torch.exp(-dist_sq / (sigma ** 2))
        base_height = self.root_states[:, 2]
        roll, pitch = self.base_euler_xyz[:, 0], self.base_euler_xyz[:, 1]
        I_unbiased = torch.ones(self.num_envs, device=self.device)
        I_unbiased = I_unbiased * (base_height > 0.5).float()
        I_unbiased = I_unbiased * (torch.abs(roll) < 0.6).float()
        I_unbiased = I_unbiased * (torch.abs(pitch) < 0.6).float()
        I_unbiased = I_unbiased * has_contact
        return base_rew * I_unbiased
