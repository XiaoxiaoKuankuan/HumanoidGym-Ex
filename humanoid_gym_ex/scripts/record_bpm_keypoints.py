# SPDX-License-Identifier: BSD-3-Clause
#
"""
从 bpm_phase_state_dataset 生成带身体关键点的 CSV。

运行命令：
    python humanoid_gym_ex/scripts/record_bpm_keypoints.py --headless

运行单个文件：
    python humanoid_gym_ex/scripts/record_bpm_keypoints.py --file bpm_070.csv --headless

运行单个文件并打开渲染：
    python humanoid_gym_ex/scripts/record_bpm_keypoints.py \
  --input_dir BPM_dance/bpm_phase_state_dataset \
  --output_dir BPM_dance/bpm_phase_state_dataset_keypoint \
  --file bpm_150.csv \
  --render

单文件只渲染检查、不保存数据：
    python humanoid_gym_ex/scripts/record_bpm_keypoints.py \
  --input_dir BPM_dance/bpm_phase_state_dataset \
  --file bpm_150.csv \
  --render \
  --no_save

常用参数：
    --input_dir bpm_phase_state_dataset
    --output_dir bpm_phase_state_dataset_keypoint
    --bpm_start 60 --bpm_end 170
    --no_save 只运行仿真/渲染检查，不写出 *_keypoint.csv

输入 CSV 格式：
    必须包含 beat_phase_rad
    必须包含 13 维 base 状态：
        base_pos_x, base_pos_y, base_pos_z
        base_quat_w, base_quat_x, base_quat_y, base_quat_z
        base_lin_vel_x, base_lin_vel_y, base_lin_vel_z
        base_ang_vel_x, base_ang_vel_y, base_ang_vel_z
    关节列按列名读取，格式为 *_pos 和对应的 *_vel。

说明：
    不要求固定 3 秒，也不要求固定帧数。现在每个 bpm 文件可以是一整个周期，
    因此不同 bpm 的行数可以不同。
    不再强制总列数等于 69；脚本按列名读取需要的 phase/base/关节列，
    额外列会忽略，缺少必要列才报错。
    输入 base 状态来自 MuJoCo 导出：地面在 FLOOR_Z（非 z=0），四元数为 w,x,y,z。
    加载到 IsaacLab/IsaacSim 时会将 root z 抬高 -FLOOR_Z。IsaacLab root/body
    四元数也是 w,x,y,z；输出 keypoint quat 会转换为 x,y,z,w，保持旧 reference
    network 数据列语义不变。

输出 CSV 顺序：
    beat_phase_rad / 关节角度 *_pos / 关节角速度 *_vel / 身体关键点信息
"""

import argparse
import os
import sys
import csv
from collections import OrderedDict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from isaaclab.app import AppLauncher
from tqdm import tqdm

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR


DATA_DT = 0.01
BPM_START = 60
BPM_END = 170
BASE_STATE_DIM = 13
TYPICAL_DOF_DIM_IN_DATA = 27
FIRST_FRAME_KEYPOINT_JUMP_THRESHOLD = 0.25

# MuJoCo 导出数据：地面高度（Isaac Gym 地面为 z=0）
MUJOCO_FLOOR_Z = -0.8834
MUJOCO_TO_ISAAC_ROOT_Z_OFFSET = -MUJOCO_FLOOR_Z  # 0.8834

JOINT_NAME_ALIASES = {
    "left_leg_pelvic_pitch": "leg_l1_joint",
    "left_leg_pelvic_roll": "leg_l2_joint",
    "left_leg_pelvic_yaw": "leg_l3_joint",
    "left_leg_knee_pitch": "leg_l4_joint",
    "left_leg_ankle_pitch": "leg_l5_joint",
    "left_leg_ankle_roll": "leg_l6_joint",
    "right_leg_pelvic_pitch": "leg_r1_joint",
    "right_leg_pelvic_roll": "leg_r2_joint",
    "right_leg_pelvic_yaw": "leg_r3_joint",
    "right_leg_knee_pitch": "leg_r4_joint",
    "right_leg_ankle_pitch": "leg_r5_joint",
    "right_leg_ankle_roll": "leg_r6_joint",
    "waist_yaw": "waist_yaw_joint",
    "left_shoulder_pitch": "upper_left_1_joint",
    "left_shoulder_roll": "upper_left_2_joint",
    "left_shoulder_yaw": "upper_left_3_joint",
    "left_elbow_pitch": "upper_left_4_joint",
    "left_wrist_yaw": "upper_left_5_joint",
    "left_wrist_pitch": "upper_left_6_joint",
    "left_wrist_roll": "upper_left_7_joint",
    "right_shoulder_pitch": "upper_right_1_joint",
    "right_shoulder_roll": "upper_right_2_joint",
    "right_shoulder_yaw": "upper_right_3_joint",
    "right_elbow_pitch": "upper_right_4_joint",
    "right_wrist_yaw": "upper_right_5_joint",
    "right_wrist_pitch": "upper_right_6_joint",
    "right_wrist_roll": "upper_right_7_joint",
}

BASE_COLUMNS = [
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
]

KEY_BODY_FIELDS = [
    ("pelvis", "base_indices"),
    ("feet", "feet_indices"),
    ("ankle", "ankle_indices"),
    ("knee", "knee_indices"),
    ("hip", "hip_indices"),
    ("pelvic_yaw", "pelvic_yaw_indices"),
    ("waist", "waist_indices"),
]


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(LEGGED_GYM_ROOT_DIR, path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Record BPM keypoint CSVs through IsaacLab/IsaacSim.")
    parser.add_argument(
        "--task",
        type=str,
        default="mrobot_music",
        choices=["mrobot_music"],
        help="Compatibility task name. This script uses the MRobot IsaacLab env directly.",
    )
    parser.add_argument(
        "--input_dir",
        default="bpm_phase_state_dataset",
        help="Directory containing bpm_070.csv ... bpm_160.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default="bpm_phase_state_dataset_keypoint",
        help="Directory for generated *_keypoint.csv files.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Run one CSV file, e.g. bpm_070.csv or /abs/path/bpm_070.csv.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Open IsaacSim viewer for visual checking. Usually used with --file.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        default=False,
        help="Run simulation/rendering only and skip writing *_keypoint.csv.",
    )
    parser.add_argument("--bpm_start", type=int, default=BPM_START)
    parser.add_argument("--bpm_end", type=int, default=BPM_END)
    parser.add_argument("--data_dt", type=float, default=DATA_DT)
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Optional BPM reference checkpoint path used only because the IsaacLab MRobot env loads it at construction.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        default=False,
        help="Skip missing bpm CSV files instead of raising an error.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Batch CSV generation should be headless by default.  --render is the
    # script-level override for visual checking.
    args.headless = not args.render
    args.input_dir = _resolve_path(args.input_dir)
    args.output_dir = _resolve_path(args.output_dir)
    if args.file is not None and not os.path.isabs(args.file):
        args.file = os.path.join(args.input_dir, args.file)
    return args


def _finite_diff_pos(pos_buffer, data_dt):
    vel_buffer = np.zeros_like(pos_buffer)
    if len(pos_buffer) > 1:
        vel_buffer[1:] = (pos_buffer[1:] - pos_buffer[:-1]) / data_dt
    return vel_buffer


def _quat_conjugate_xyzw(quat):
    quat_conj = quat.copy()
    quat_conj[..., :3] *= -1.0
    return quat_conj


def _quat_multiply_xyzw(quat_a, quat_b):
    ax, ay, az, aw = np.moveaxis(quat_a, -1, 0)
    bx, by, bz, bw = np.moveaxis(quat_b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _finite_diff_quat_ang_vel(quat_buffer, data_dt):
    ang_vel_buffer = np.zeros(quat_buffer.shape[:-1] + (3,), dtype=quat_buffer.dtype)
    if len(quat_buffer) <= 1:
        return ang_vel_buffer

    quat_prev = quat_buffer[:-1]
    quat_curr = quat_buffer[1:]
    quat_prev = quat_prev / np.maximum(np.linalg.norm(quat_prev, axis=-1, keepdims=True), 1e-8)
    quat_curr = quat_curr / np.maximum(np.linalg.norm(quat_curr, axis=-1, keepdims=True), 1e-8)

    same_hemi = np.sum(quat_prev * quat_curr, axis=-1, keepdims=True) >= 0.0
    quat_curr = np.where(same_hemi, quat_curr, -quat_curr)

    quat_delta = _quat_multiply_xyzw(quat_curr, _quat_conjugate_xyzw(quat_prev))
    quat_delta = quat_delta / np.maximum(np.linalg.norm(quat_delta, axis=-1, keepdims=True), 1e-8)

    rot_vec = quat_delta[..., :3]
    rot_w = np.clip(quat_delta[..., 3], -1.0, 1.0)
    rot_norm = np.linalg.norm(rot_vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(rot_norm, rot_w[..., None])
    axis = np.divide(rot_vec, rot_norm, out=np.zeros_like(rot_vec), where=rot_norm > 1e-8)
    ang_vel_buffer[1:] = axis * angle / data_dt
    return ang_vel_buffer


def _repair_first_keypoint_frame_if_needed(buffers, jump_threshold=FIRST_FRAME_KEYPOINT_JUMP_THRESHOLD):
    if not buffers or next(iter(buffers.values())).shape[0] < 2:
        return

    max_jump = 0.0
    for field_name, _ in KEY_BODY_FIELDS:
        pos = buffers[f"{field_name}_pos"]
        max_jump = max(max_jump, float(np.linalg.norm(pos[1] - pos[0], axis=-1).max()))

    if max_jump <= jump_threshold:
        return

    print(
        "[record_bpm_keypoints] repaired first keypoint frame: "
        f"max initial body jump={max_jump:.6f}m"
    )
    for field_name, _ in KEY_BODY_FIELDS:
        buffers[f"{field_name}_pos"][0] = buffers[f"{field_name}_pos"][1]
        buffers[f"{field_name}_quat"][0] = buffers[f"{field_name}_quat"][1]
        buffers[f"{field_name}_vel"][0:2] = 0.0
        buffers[f"{field_name}_ang_vel"][0:2] = 0.0


def _configure_env(args):
    from humanoid_gym_ex.envs.robots.mrobot.isaaclab_env import (
        MrobotMimicBPMIsaacLabEnv,
        MrobotMimicBPMIsaacLabEnvCfg,
    )

    env_cfg = MrobotMimicBPMIsaacLabEnvCfg()
    env_cfg.seed = 123145
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.disable_domain_randomization = True
    env_cfg.deterministic_reset = True
    env_cfg.profile_step_timings = False
    if args.reference_model is not None:
        env_cfg.reference_model_path = args.reference_model

    env = MrobotMimicBPMIsaacLabEnv(env_cfg)
    env.reset()
    return env


def _read_bpm_csv(csv_path):
    df = pd.read_csv(csv_path)
    columns = list(df.columns)

    required_columns = ["beat_phase_rad"] + BASE_COLUMNS
    missing_required = [name for name in required_columns if name not in df.columns]
    if missing_required:
        raise ValueError(f"{csv_path} missing required columns: {missing_required}")

    dof_pos_columns = [name for name in columns if name.endswith("_pos")]
    raw_dof_vel_columns = [name for name in columns if name.endswith("_vel")]
    vel_by_joint = {_normalize_joint_name(name): name for name in raw_dof_vel_columns}

    dof_vel_columns = []
    missing_vel = []
    for pos_name in dof_pos_columns:
        joint_name = _normalize_joint_name(pos_name)
        vel_name = vel_by_joint.get(joint_name)
        if vel_name is None:
            missing_vel.append(joint_name)
        else:
            dof_vel_columns.append(vel_name)

    if not dof_pos_columns:
        raise ValueError(f"{csv_path} has no *_pos joint columns")
    if missing_vel:
        raise ValueError(f"{csv_path} missing *_vel columns for joints: {missing_vel}")

    if len(dof_pos_columns) != TYPICAL_DOF_DIM_IN_DATA:
        print(
            f"[record_bpm_keypoints] {os.path.basename(csv_path)} has "
            f"{len(dof_pos_columns)} joint columns, typical is {TYPICAL_DOF_DIM_IN_DATA}; "
            "will map available joints by name/order."
        )

    phase = df[["beat_phase_rad"]].to_numpy(dtype=np.float32)
    root = df[BASE_COLUMNS].to_numpy(dtype=np.float32)
    dof_pos = df[dof_pos_columns].to_numpy(dtype=np.float32)
    dof_vel = df[dof_vel_columns].to_numpy(dtype=np.float32)
    return phase, root, dof_pos, dof_vel, dof_pos_columns, dof_vel_columns


def _normalize_joint_name(name):
    for suffix in ("_pos", "_vel"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _build_dof_mapping(env, dof_pos_columns):
    env_dof_names = list(env.robot.joint_names)
    env_num_dof = int(env.robot.num_joints)
    env_name_to_idx = {_normalize_joint_name(name): idx for idx, name in enumerate(env_dof_names)}
    mapping = []
    mapped_by_name = 0

    for data_idx, column in enumerate(dof_pos_columns):
        joint_name = _normalize_joint_name(column)
        env_joint_name = JOINT_NAME_ALIASES.get(joint_name, joint_name)
        if env_joint_name in env_name_to_idx:
            env_idx = env_name_to_idx[env_joint_name]
            mapped_by_name += 1
        elif data_idx < env_num_dof:
            env_idx = data_idx
        else:
            raise ValueError(f"Cannot map input joint column '{column}' to env DOF")
        mapping.append((data_idx, env_idx))

    if mapped_by_name != len(dof_pos_columns):
        print(
            f"[record_bpm_keypoints] only {mapped_by_name}/{len(dof_pos_columns)} joints "
            "matched by name; remaining joints use input order."
        )
    if env_num_dof > len(dof_pos_columns):
        print(
            f"[record_bpm_keypoints] input has {len(dof_pos_columns)} joints, env has {env_num_dof}; "
            "unmapped DOFs keep default positions and zero velocities."
        )
    return mapping


def _default_dof_pos(env):
    return env.robot.data.default_joint_pos[0].clone()


def _make_root_states(root_np, device):
    """MuJoCo CSV base 状态 -> IsaacLab root state (13, quat=wxyz)."""
    root_states = torch.zeros(root_np.shape[0], 13, dtype=torch.float, device=device)
    root_tensor = torch.from_numpy(root_np).to(device=device, dtype=torch.float)

    # pos: MuJoCo 地面在 MUJOCO_FLOOR_Z，Isaac 地面在 z=0
    root_states[:, 0:3] = root_tensor[:, 0:3]
    root_states[:, 2] += MUJOCO_TO_ISAAC_ROOT_Z_OFFSET

    # quat: CSV 和 IsaacLab 都为 w,x,y,z
    root_states[:, 3:7] = root_tensor[:, 3:7]

    root_states[:, 7:10] = root_tensor[:, 7:10]
    root_states[:, 10:13] = root_tensor[:, 10:13]
    return root_states


def _quat_wxyz_to_xyzw_np(quat):
    return np.concatenate((quat[..., 1:4], quat[..., 0:1]), axis=-1)


def _collect_keypoints_for_file(env, phase, root_np, dof_pos_np, dof_vel_np, dof_mapping, data_dt, render=False):
    num_frames = phase.shape[0]
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    root_states_all = _make_root_states(root_np, env.device)
    default_dof_pos = _default_dof_pos(env)
    num_dof = int(env.robot.num_joints)

    buffers = OrderedDict()
    for field_name, _ in KEY_BODY_FIELDS:
        buffers[f"{field_name}_pos"] = []
        buffers[f"{field_name}_vel"] = []
        buffers[f"{field_name}_quat"] = []
        buffers[f"{field_name}_ang_vel"] = []

    for frame_idx in tqdm(range(num_frames), desc="frames", leave=False):
        dof_pos = default_dof_pos.repeat(env.num_envs, 1)
        dof_vel = torch.zeros(env.num_envs, num_dof, dtype=torch.float, device=env.device)
        dof_pos_src = torch.from_numpy(dof_pos_np[frame_idx]).to(env.device, dtype=torch.float)
        dof_vel_src = torch.from_numpy(dof_vel_np[frame_idx]).to(env.device, dtype=torch.float)

        for data_idx, env_idx in dof_mapping:
            dof_pos[:, env_idx] = dof_pos_src[data_idx]
            dof_vel[:, env_idx] = dof_vel_src[data_idx]

        root_states = root_states_all[frame_idx : frame_idx + 1].repeat(env.num_envs, 1)

        env.robot.write_root_pose_to_sim(root_states[:, :7], env_ids)
        env.robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)
        env.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)

        for field_name, indices_attr in KEY_BODY_FIELDS:
            indices = getattr(env, indices_attr)
            rigid_state = env.robot.data.body_state_w[:, indices]
            buffers[f"{field_name}_pos"].append(rigid_state[0, :, 0:3].cpu().numpy())
            buffers[f"{field_name}_vel"].append(rigid_state[0, :, 7:10].cpu().numpy())
            # IsaacLab body_state_w stores quat as w,x,y,z.  The keypoint CSV
            # headers and reference-network loader expect x,y,z,w.
            buffers[f"{field_name}_quat"].append(_quat_wxyz_to_xyzw_np(rigid_state[0, :, 3:7].cpu().numpy()))
            buffers[f"{field_name}_ang_vel"].append(rigid_state[0, :, 10:13].cpu().numpy())

        if render:
            env.sim.render()

    for key, value in buffers.items():
        buffers[key] = np.asarray(value, dtype=np.float32)

    # As in record_refpos.py, derive key-body velocities from exported poses.
    # This avoids set_state cache spikes and keeps velocity consistent with pose.
    for field_name, _ in KEY_BODY_FIELDS:
        buffers[f"{field_name}_vel"] = _finite_diff_pos(buffers[f"{field_name}_pos"], data_dt)
        buffers[f"{field_name}_ang_vel"] = _finite_diff_quat_ang_vel(buffers[f"{field_name}_quat"], data_dt)

    _repair_first_keypoint_frame_if_needed(buffers)

    return buffers


def _field_headers(field_name, field_sample):
    arr = np.asarray(field_sample)
    if arr.ndim == 0:
        return [field_name]
    if arr.ndim == 1:
        return [f"{field_name}_{i}" for i in range(arr.shape[0])]

    if arr.shape[-1] == 3:
        last_dim_names = ["x", "y", "z"]
    elif arr.shape[-1] == 4:
        last_dim_names = ["x", "y", "z", "w"]
    else:
        last_dim_names = [str(i) for i in range(arr.shape[-1])]

    headers = []
    for prefix_idx in np.ndindex(*arr.shape[:-1]):
        prefix = field_name + "".join([f"_{idx}" for idx in prefix_idx])
        headers.extend([f"{prefix}_{name}" for name in last_dim_names])
    return headers


def _write_output_csv(csv_path, phase, dof_pos, dof_vel, dof_pos_columns, dof_vel_columns, keypoint_buffers):
    headers = ["beat_phase_rad"] + list(dof_pos_columns) + list(dof_vel_columns)
    for field, values in keypoint_buffers.items():
        headers.extend(_field_headers(field, values[0]))

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for frame_idx in range(phase.shape[0]):
            row = []
            row.extend(phase[frame_idx].reshape(-1).tolist())
            row.extend(dof_pos[frame_idx].reshape(-1).tolist())
            row.extend(dof_vel[frame_idx].reshape(-1).tolist())
            for values in keypoint_buffers.values():
                row.extend(values[frame_idx].reshape(-1).tolist())
            writer.writerow(row)


def record_bpm_keypoints(args):
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.file is not None:
        input_paths = [args.file]
        missing = [path for path in input_paths if not os.path.isfile(path)]
    else:
        file_names = [f"bpm_{bpm:03d}.csv" for bpm in range(args.bpm_start, args.bpm_end + 1)]
        input_paths = [os.path.join(args.input_dir, name) for name in file_names]
        missing = [path for path in input_paths if not os.path.isfile(path)]

    if missing and not args.allow_missing:
        preview = ", ".join([os.path.basename(path) for path in missing[:5]])
        raise FileNotFoundError(f"Missing {len(missing)} bpm CSV files; first missing: {preview}")

    env = _configure_env(args)
    dof_mapping_cache = {}
    try:
        for input_path in tqdm(input_paths, desc="bpm files"):
            if not os.path.isfile(input_path):
                continue

            file_name = os.path.basename(input_path)
            phase, root, dof_pos, dof_vel, dof_pos_columns, dof_vel_columns = _read_bpm_csv(input_path)
            mapping_key = tuple(dof_pos_columns)
            if mapping_key not in dof_mapping_cache:
                dof_mapping_cache[mapping_key] = _build_dof_mapping(env, dof_pos_columns)
            dof_mapping = dof_mapping_cache[mapping_key]

            keypoint_buffers = _collect_keypoints_for_file(
                env,
                phase,
                root,
                dof_pos,
                dof_vel,
                dof_mapping,
                args.data_dt,
                render=args.render,
            )
            if args.no_save:
                print(f"[record_bpm_keypoints] no_save enabled; skipped writing: {file_name}")
                continue

            output_path = os.path.join(args.output_dir, file_name.replace(".csv", "_keypoint.csv"))
            _write_output_csv(output_path, phase, dof_pos, dof_vel, dof_pos_columns, dof_vel_columns, keypoint_buffers)
            print(f"[record_bpm_keypoints] saved: {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    args_cli = _parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        record_bpm_keypoints(args_cli)
    finally:
        simulation_app.close()
