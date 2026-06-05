"""Shared MRobot trajectory-reference loader for dance mimic tasks."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR


DEFAULT_DANCE_MOTION_FILES = [
    "ref_pos/JT_GuangHuiSuiYue_DongTai_50HZ_keypoint.npz",
]

REQUIRED_TRAJECTORY_FIELDS = (
    "dof_pos",
    "dof_vel",
    "root_states",
    "root_linvel",
    "root_angvel",
    "euler_xyz",
    "foot_height",
    "feet_contact",
    "pelvis_pos",
    "pelvis_vel",
    "pelvis_quat",
    "pelvis_ang_vel",
    "feet_pos",
    "feet_vel",
    "feet_quat",
    "feet_ang_vel",
    "knee_pos",
    "knee_vel",
    "knee_quat",
    "knee_ang_vel",
    "hip_pos",
    "hip_vel",
    "hip_quat",
    "hip_ang_vel",
    "pelvic_yaw_pos",
    "pelvic_yaw_vel",
    "pelvic_yaw_quat",
    "pelvic_yaw_ang_vel",
    "waist_pos",
    "waist_vel",
    "waist_quat",
    "waist_ang_vel",
)


def parse_motion_files(value):
    """Return a clean list from list/tuple/comma-separated motion path config."""
    if value is None:
        return list(DEFAULT_DANCE_MOTION_FILES)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    files = []
    for item in value:
        if isinstance(item, str) and "," in item:
            files.extend(parse_motion_files(item))
        else:
            files.append(str(item))
    return [item for item in files if item]


def resolve_motion_file(path):
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(LEGGED_GYM_ROOT_DIR, path),
        os.path.join(LEGGED_GYM_ROOT_DIR, "humanoid_gym_ex", path),
        os.path.join("/home/weil/hl_rl/hl_rl/humanoid", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def get_motion_files_from_cfg(motion_cfg):
    return [resolve_motion_file(path) for path in parse_motion_files(getattr(motion_cfg, "files", None))]


def _pad_motion_tensor(motion_data, key, max_length, device, allow_legacy_keypoint_fallback=False):
    if key not in motion_data:
        if not allow_legacy_keypoint_fallback:
            raise KeyError(key)
        if key == "pelvis_pos":
            return _pad_motion_tensor(motion_data, "root_states", max_length, device)[:, :3].unsqueeze(1)
        if key == "pelvis_vel":
            return _pad_motion_tensor(motion_data, "root_states", max_length, device)[:, 7:10].unsqueeze(1)
        if key == "pelvis_quat":
            return _pad_motion_tensor(motion_data, "root_states", max_length, device)[:, 3:7].unsqueeze(1)
        if key == "pelvis_ang_vel":
            return _pad_motion_tensor(motion_data, "root_states", max_length, device)[:, 10:13].unsqueeze(1)
        fallback_map = {
            "pelvic_yaw_pos": "hip_pos",
            "pelvic_yaw_vel": "hip_vel",
            "pelvic_yaw_quat": "hip_quat",
            "pelvic_yaw_ang_vel": "hip_ang_vel",
        }
        if key in fallback_map:
            return _pad_motion_tensor(motion_data, fallback_map[key], max_length, device)
        raise KeyError(key)
    tensor = torch.from_numpy(np.asarray(motion_data[key])).to(device=device, dtype=torch.float32)
    pad_len = max_length - tensor.shape[0]
    if pad_len <= 0:
        return tensor
    pad_frame = tensor[-1:].repeat(pad_len, *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, pad_frame), dim=0)


def load_mrobot_trajectory_library(
    motion_files,
    device,
    required_fields=REQUIRED_TRAJECTORY_FIELDS,
    allow_legacy_keypoint_fallback=False,
):
    resolved_files = [resolve_motion_file(path) for path in parse_motion_files(motion_files)]
    if not resolved_files:
        raise ValueError("MRobot dance motion files are empty.")

    raw_motions = []
    motion_lengths = []
    for path in resolved_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"MRobot dance motion file not found: {path}")
        motion = dict(np.load(path, allow_pickle=True))
        missing = [key for key in required_fields if key not in motion]
        if missing and not allow_legacy_keypoint_fallback:
            raise KeyError(f"Motion file {path} missing required fields: {missing}")
        if "dof_pos" not in motion:
            raise KeyError(f"Motion file {path} missing dof_pos.")
        if motion["dof_pos"].ndim != 2 or motion["dof_pos"].shape[1] != 29:
            raise ValueError(f"Motion file {path} dof_pos must have shape [T, 29], got {motion['dof_pos'].shape}.")
        raw_motions.append(motion)
        motion_lengths.append(int(motion["dof_pos"].shape[0]))

    max_length = max(motion_lengths)
    buffers = {}
    for key in required_fields:
        tensors = [
            _pad_motion_tensor(
                motion,
                key,
                max_length,
                device,
                allow_legacy_keypoint_fallback=allow_legacy_keypoint_fallback,
            )
            for motion in raw_motions
        ]
        buffers[key] = torch.stack(tensors, dim=0)

    return SimpleNamespace(
        files=resolved_files,
        data_length=len(resolved_files),
        demo_length=max_length,
        demo_lengths=torch.tensor(motion_lengths, device=device, dtype=torch.long),
        buffers=buffers,
    )
