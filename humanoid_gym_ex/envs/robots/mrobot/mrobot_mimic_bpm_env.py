"""IsaacGym MRobot BPM/music mimic task.

This file owns the reference-network/BPM logic.  The shared robot action,
observation, reward, reset helpers live in ``mrobot_mimic_common_env.py``.
"""

import os

import numpy as np
import torch
from isaacgym.torch_utils import *

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.robots.mrobot.mrobot_legged_robot import get_euler_xyz_tensor
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_env import MrobotMimicCommonEnv
from humanoid_gym_ex.utils import torch_utils
from humanoid_gym_ex.utils.reference_state import (
    JOINT_NAME_ALIASES,
    ReferenceStateNet,
    encode_bpm_phase,
)


class MrobotMimicBPMEnv(MrobotMimicCommonEnv):
    """BPM/music reference-network mimic task."""

    def _init_task_reference(self):
        self._init_reference_network()
        self._init_reference_command_buffers()
        self._init_reference_state_buffers()
        self.data_length = 1
        self.demo_length = max(1, int(round(self.max_episode_length_s / self.dt)))
        self.demo_lengths = torch.full((1,), self.demo_length, device=self.device, dtype=torch.long)
        print("[mrobot_bpm] BPM reference network loaded")
        print(f"[mrobot_bpm] reference model: {self.reference_model_path}")
        print(
            "[mrobot_bpm] Observation layout changed: actor obs 64 -> 76. "
            "Old checkpoints and normalizer statistics are incompatible. "
            "Train from scratch or reset normalizer.",
            flush=True,
        )

    def _resolve_reference_model_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(LEGGED_GYM_ROOT_DIR, path)

    def _init_reference_network(self):
        motion_cfg = getattr(self.cfg, "motion", None)
        model_path = getattr(motion_cfg, "reference_model_path", "deploy/reference_state_keypoint_model.pt")
        self.reference_model_path = self._resolve_reference_model_path(model_path)
        if not os.path.exists(self.reference_model_path):
            raise FileNotFoundError(
                "BPM reference model checkpoint not found: "
                f"{self.reference_model_path}. Set cfg.motion.reference_model_path "
                "or pass an absolute path; datasets and checkpoints are intentionally not bundled."
            )
        checkpoint = torch.load(self.reference_model_path, map_location=self.device, weights_only=False)
        self.reference_output_columns = list(checkpoint["output_columns"])
        self.reference_column_index = {name: idx for idx, name in enumerate(self.reference_output_columns)}
        self.reference_bpm_mean = torch.tensor(float(checkpoint["bpm_mean"]), device=self.device)
        self.reference_bpm_std = torch.tensor(float(checkpoint["bpm_std"]), device=self.device).clamp(min=1e-6)
        self.reference_target_mean = torch.as_tensor(checkpoint["target_mean"], device=self.device, dtype=torch.float32)
        self.reference_target_std = torch.as_tensor(checkpoint["target_std"], device=self.device, dtype=torch.float32)
        self.reference_net = ReferenceStateNet(
            int(checkpoint["input_dim"]),
            int(checkpoint["output_dim"]),
            checkpoint["hidden"],
        ).to(self.device)
        self.reference_net.load_state_dict(checkpoint["model_state_dict"])
        self.reference_net.eval()
        for param in self.reference_net.parameters():
            param.requires_grad_(False)

        self.ref_dof_pos_indices, self.ref_dof_pos_mask = self._build_dof_column_indices("_pos")
        self.ref_dof_vel_indices, self.ref_dof_vel_mask = self._build_dof_column_indices("_vel")

    def _build_dof_column_indices(self, suffix):
        reverse_alias = {env_name: data_name for data_name, env_name in JOINT_NAME_ALIASES.items()}
        indices = []
        mask = []
        for dof_name in self.dof_names:
            base_name = reverse_alias.get(dof_name, dof_name)
            if base_name.endswith("_joint"):
                base_name = base_name[: -len("_joint")]
            column_name = base_name + suffix
            col_idx = self.reference_column_index.get(column_name, -1)
            indices.append(max(col_idx, 0))
            mask.append(col_idx >= 0)
        return (
            torch.tensor(indices, device=self.device, dtype=torch.long),
            torch.tensor(mask, device=self.device, dtype=torch.bool),
        )

    def _init_reference_command_buffers(self):
        motion_cfg = getattr(self.cfg, "motion", None)
        bpm_range = getattr(motion_cfg, "bpm_range", [70.0, 160.0])
        self.reference_bpm_min = float(bpm_range[0])
        self.reference_bpm_max = float(bpm_range[1])
        self.reference_include_zero_bpm = bool(getattr(motion_cfg, "include_zero_bpm", False))
        self.reference_sample_integer_bpm = bool(getattr(motion_cfg, "sample_integer_bpm", False))
        self.reference_fixed_bpm = getattr(motion_cfg, "fixed_bpm", None)
        self.randomize_init_phase = bool(getattr(motion_cfg, "randomize_init_phase", True))
        self.init_phase_range = getattr(motion_cfg, "init_phase_range", [0.0, 2.0 * np.pi])
        self.foot_contact_height_threshold = float(getattr(motion_cfg, "foot_contact_height_threshold", 0.08))
        self.bpm_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase_rad = torch.zeros(self.num_envs, 1, device=self.device)
        self.normalized_bpm_cmd = torch.zeros(self.num_envs, 1, device=self.device)

    def _predict_reference_state(self):
        encoded = encode_bpm_phase(
            self.bpm_cmd,
            self.phase_rad,
            self.reference_bpm_mean,
            self.reference_bpm_std,
        )
        with torch.no_grad():
            pred_norm = self.reference_net(encoded)
        return pred_norm * self.reference_target_std + self.reference_target_mean

    def _extract_ref_field(self, pred, field_name, num_parts, width, kind):
        axes = ["x", "y", "z"] if width == 3 else ["x", "y", "z", "w"]
        values = torch.zeros(self.num_envs, num_parts, width, device=self.device)
        if width == 4:
            values[..., 3] = 1.0
        for part_idx in range(num_parts):
            for axis_idx, axis in enumerate(axes):
                col = f"{field_name}_{kind}_{part_idx}_{axis}"
                if col in self.reference_column_index:
                    values[:, part_idx, axis_idx] = pred[:, self.reference_column_index[col]]
        return values

    def _split_reference_prediction(self, pred):
        dof_pos_pred = pred[:, self.ref_dof_pos_indices]
        dof_vel_pred = pred[:, self.ref_dof_vel_indices]
        self.ref_dof_pos[:] = self.default_dof_pos
        self.ref_dof_vel.zero_()
        self.ref_dof_pos[:, self.ref_dof_pos_mask] = dof_pos_pred[:, self.ref_dof_pos_mask]
        self.ref_dof_vel[:, self.ref_dof_vel_mask] = dof_vel_pred[:, self.ref_dof_vel_mask]

        self.ref_pelvis_pos = self._extract_ref_field(pred, "pelvis", 1, 3, "pos")
        self.ref_pelvis_vel = self._extract_ref_field(pred, "pelvis", 1, 3, "vel")
        self.ref_pelvis_quat = self._extract_ref_field(pred, "pelvis", 1, 4, "quat")
        self.ref_pelvis_ang_vel = self._extract_ref_field(pred, "pelvis", 1, 3, "ang_vel")
        self.ref_feet_pos = self._extract_ref_field(pred, "feet", 2, 3, "pos")
        self.ref_feet_vel = self._extract_ref_field(pred, "feet", 2, 3, "vel")
        self.ref_feet_quat = self._extract_ref_field(pred, "feet", 2, 4, "quat")
        self.ref_feet_ang_vel = self._extract_ref_field(pred, "feet", 2, 3, "ang_vel")
        self.ref_knee_pos = self._extract_ref_field(pred, "knee", 2, 3, "pos")
        self.ref_knee_vel = self._extract_ref_field(pred, "knee", 2, 3, "vel")
        self.ref_knee_quat = self._extract_ref_field(pred, "knee", 2, 4, "quat")
        self.ref_knee_ang_vel = self._extract_ref_field(pred, "knee", 2, 3, "ang_vel")
        self.ref_hip_pos = self._extract_ref_field(pred, "hip", 2, 3, "pos")
        self.ref_hip_vel = self._extract_ref_field(pred, "hip", 2, 3, "vel")
        self.ref_hip_quat = self._extract_ref_field(pred, "hip", 2, 4, "quat")
        self.ref_hip_ang_vel = self._extract_ref_field(pred, "hip", 2, 3, "ang_vel")
        self.ref_pelvic_yaw_pos = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "pos")
        self.ref_pelvic_yaw_vel = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "vel")
        self.ref_pelvic_yaw_quat = self._extract_ref_field(pred, "pelvic_yaw", 2, 4, "quat")
        self.ref_pelvic_yaw_ang_vel = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "ang_vel")
        self.ref_waist_pos = self._extract_ref_field(pred, "waist", 1, 3, "pos")
        self.ref_waist_vel = self._extract_ref_field(pred, "waist", 1, 3, "vel")
        self.ref_waist_quat = self._extract_ref_field(pred, "waist", 1, 4, "quat")
        self.ref_waist_ang_vel = self._extract_ref_field(pred, "waist", 1, 3, "ang_vel")

        self.ref_feet_quat = self.ref_feet_quat / self.ref_feet_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.ref_knee_quat = self.ref_knee_quat / self.ref_knee_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.ref_hip_quat = self.ref_hip_quat / self.ref_hip_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.ref_pelvis_quat = self.ref_pelvis_quat / self.ref_pelvis_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.ref_pelvic_yaw_quat = self.ref_pelvic_yaw_quat / self.ref_pelvic_yaw_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.ref_waist_quat = self.ref_waist_quat / self.ref_waist_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        self.ref_foot_height = self.ref_feet_pos[..., 2]
        self.ref_feet_contact = (self.ref_foot_height < self.foot_contact_height_threshold).float()
        self.ref_root_linvel = self.ref_waist_vel[:, 0, :]
        self.ref_root_angvel = self.ref_waist_ang_vel[:, 0, :]
        self.ref_euler_xyz = get_euler_xyz_tensor(self.ref_waist_quat[:, 0, :])

    def compute_ref_state(self):
        pred = self._predict_reference_state()
        self._split_reference_prediction(pred)

    def _get_actor_reference_extra_obs(self):
        return torch.cat(
            (
                torch.sin(self.phase_rad),
                torch.cos(self.phase_rad),
                self.normalized_bpm_cmd,
            ),
            dim=-1,
        )

    def _get_privileged_reference_phase_obs(self, norm_phase):
        return norm_phase

    def _get_reference_norm_phase(self):
        return self.phase_rad / (2.0 * torch.pi)

    def _advance_reference_phase(self):
        self.phase_rad[:] = torch.remainder(
            self.phase_rad + (2.0 * torch.pi * self.bpm_cmd / 60.0) * self.dt,
            2.0 * torch.pi,
        )

    def _get_noncontrolled_ref_actions(self):
        self.compute_ref_state()
        ref_pos = self.ref_dof_pos[:, self.ref_num_notcontrol]
        return ref_pos / self.cfg.control.action_scale

    def _resample_reference_commands(self, env_ids):
        if self.reference_fixed_bpm is None:
            if self.reference_sample_integer_bpm:
                bpm_min = int(round(self.reference_bpm_min))
                bpm_max = int(round(self.reference_bpm_max))
                if bpm_max < bpm_min:
                    raise ValueError(f"Invalid bpm_range: [{self.reference_bpm_min}, {self.reference_bpm_max}]")
                num_regular_bpms = bpm_max - bpm_min + 1
                if self.reference_include_zero_bpm:
                    bpm_choice = torch.randint(
                        0,
                        num_regular_bpms + 1,
                        (len(env_ids), 1),
                        device=self.device,
                    )
                    sampled_bpm = torch.where(
                        bpm_choice == 0,
                        torch.zeros_like(bpm_choice),
                        bpm_choice + bpm_min - 1,
                    )
                else:
                    sampled_bpm = torch.randint(
                        bpm_min,
                        bpm_max + 1,
                        (len(env_ids), 1),
                        device=self.device,
                    )
                self.bpm_cmd[env_ids] = sampled_bpm.to(dtype=self.bpm_cmd.dtype)
            else:
                self.bpm_cmd[env_ids] = torch_rand_float(
                    self.reference_bpm_min,
                    self.reference_bpm_max,
                    (len(env_ids), 1),
                    device=self.device,
                )
                if self.reference_include_zero_bpm:
                    zero_mask = torch.rand((len(env_ids), 1), device=self.device) < 0.5
                    self.bpm_cmd[env_ids] = torch.where(
                        zero_mask,
                        torch.zeros_like(self.bpm_cmd[env_ids]),
                        self.bpm_cmd[env_ids],
                    )
        else:
            self.bpm_cmd[env_ids] = float(self.reference_fixed_bpm)
        self.normalized_bpm_cmd[env_ids] = (self.bpm_cmd[env_ids] - self.reference_bpm_mean) / self.reference_bpm_std
        if self.randomize_init_phase:
            self.phase_rad[env_ids] = torch_rand_float(
                float(self.init_phase_range[0]),
                float(self.init_phase_range[1]),
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            self.phase_rad[env_ids] = 0.0
        self.episode_phase_buf[env_ids] = 0

    def _reward_imition_root_yaw(self):
        heading_rot = torch_utils.calc_heading_quat_inv(self.base_quat)
        rot_ = quat_mul(heading_rot, self.last_root_quat)
        rot_yaw = torch.square(get_euler_xyz_tensor(rot_)[:, 2])

        cycle_start_mask = self.phase_rad[:, 0] < (2.0 * torch.pi * self.bpm_cmd[:, 0] / 60.0) * self.dt
        self.last_root_quat[cycle_start_mask] = self.root_states[cycle_start_mask, 3:7]

        return torch.exp(-8.0 * rot_yaw)


__all__ = ["MrobotMimicBPMEnv"]
