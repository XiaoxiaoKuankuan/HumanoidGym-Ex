"""Generate MRobot dance keypoint NPZ/CSV with IsaacLab.

This is the IsaacLab replacement for the old IsaacGym
``/home/weil/hl_rl/hl_rl/humanoid/scripts/record_refpos.py``.  It reads the
old dance ``.data`` format, writes robot states into IsaacLab/IsaacSim, samples
body keypoints, and saves both ``*_keypoint.npz`` and ``*_keypoint.csv`` from
the exact same ordered dictionary so the two files are numerically consistent.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR


PHYSICS_DT = 0.005
CONTROL_DT = 0.02
CONTROL_DECIMATION = 4
DATA_DT = CONTROL_DT
DATA_STRIDE = 2
CLEAN_INITIAL_COPY_SOURCE_FRAME = 2
DEFAULT_INPUT_DIR = "/home/weil/HumanoidGym-Ex/ref_pos"
DEFAULT_OUTPUT_DIR = "/home/weil/HumanoidGym-Ex/ref_pos"
DEFAULT_FILES = [
    "yellowdongtai_casbot02-guitar2_6R7R2R1R_hand01_whole_100hz_v1.data",
    "yellowdongtai_casbot02-bass1_6R7R2R1R_hand01_whole_100hz_v1.data",
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
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.join(LEGGED_GYM_ROOT_DIR, path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Record MRobot dance keypoint NPZ/CSV through IsaacLab.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory containing old .data dance files.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated *_keypoint.npz/csv.")
    parser.add_argument("--file", type=str, default=None, help="Single .data file name or absolute path.")
    parser.add_argument("--data_dt", type=float, default=DATA_DT, help="Output keypoint frame period. Default 0.02s = 50Hz.")
    parser.add_argument(
        "--data_stride",
        type=int,
        default=DATA_STRIDE,
        help="Input .data frame stride. Default 2 converts old 100Hz .data to 50Hz keypoints.",
    )
    parser.add_argument("--render", action="store_true", help="Open viewer while recording.")
    parser.add_argument("--no_save", action="store_true", help="Run conversion without writing output files.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing default files.")
    parser.add_argument("--prepend_stand_s", type=float, default=0.0, help="Prepend still frames after recording.")
    parser.add_argument("--append_stand_s", type=float, default=0.0, help="Append still frames after recording.")
    parser.add_argument(
        "--contact_height_threshold",
        type=float,
        default=0.095,
        help="Foot world-z threshold for height+velocity contact generation.",
    )
    parser.add_argument(
        "--contact_velocity_threshold",
        type=float,
        default=0.25,
        help="Foot vertical velocity threshold for height+velocity contact generation.",
    )
    parser.add_argument(
        "--contact_smooth_min_len",
        type=int,
        default=2,
        help="Minimum binary contact run length; shorter 0/1 glitches are smoothed.",
    )
    parser.add_argument(
        "--contact_force_threshold",
        type=float,
        default=1.0,
        help="Deprecated/unused compatibility option. Contact is no longer generated from IsaacLab force.",
    )
    parser.add_argument(
        "--show_cpu_governor_warning",
        action="store_true",
        help="Show IsaacSim's harmless missing cpufreq/scaling_governor startup warning.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = not args.render
    args.input_dir = _resolve_path(args.input_dir)
    args.output_dir = _resolve_path(args.output_dir)
    if args.file is not None:
        file_arg = os.path.expanduser(args.file)
        if os.path.isabs(file_arg):
            args.file = file_arg
        else:
            candidates = [
                os.path.join(LEGGED_GYM_ROOT_DIR, file_arg),
                os.path.join(args.input_dir, file_arg),
                os.path.join(args.input_dir, os.path.basename(file_arg)),
            ]
            args.file = next((path for path in candidates if os.path.isfile(path)), candidates[0])
    return args


def _install_cpu_governor_stderr_filter(enabled=True):
    """Hide one known harmless IsaacSim startup line on servers without cpufreq.

    IsaacSim/Kit prints a raw shell ``cat`` error when Linux does not expose
    ``/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor``.  That is common
    on virtualized EPYC servers and does not affect simulation.  Filter only
    that exact startup noise at the file-descriptor level so real stderr output
    and tracebacks still pass through.
    """
    if not enabled or os.environ.get("HUMANOID_GYM_SHOW_CPU_GOVERNOR_WARNING") == "1":
        return None

    try:
        original_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return None

    needle = b"/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

    def _forward_stderr():
        pending = b""
        with os.fdopen(read_fd, "rb", buffering=0) as source:
            while True:
                chunk = source.read(4096)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if needle not in line:
                        os.write(original_stderr_fd, line + b"\n")
            if pending and needle not in pending:
                os.write(original_stderr_fd, pending)
        os.close(original_stderr_fd)

    thread = threading.Thread(target=_forward_stderr, daemon=True)
    thread.start()
    return thread


def _read_data(path):
    rows = []
    with open(path, "r") as file:
        for line in file:
            fields = line.strip().split(",")
            if fields and fields[-1] == "":
                fields = fields[:-1]
            if not fields:
                continue
            try:
                rows.append([float(item) for item in fields])
            except ValueError:
                continue
    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 61:
        raise ValueError(f"{path} must contain at least 61 numeric columns, got {data.shape}")
    print(f"[record_dance_keypoints_lab] loaded {path}: {data.shape}", flush=True)
    return data


def _data_row_to_canonical_state(row):
    dof_pos = np.zeros(29, dtype=np.float32)
    dof_pos[0:6] = row[12:18]
    dof_pos[6:12] = row[18:24]
    dof_pos[12] = row[60]
    dof_pos[13:20] = row[24:31]
    dof_pos[20:27] = row[31:38]
    dof_pos[27] = row[58]
    dof_pos[28] = row[59]

    root_xyzw = np.zeros(13, dtype=np.float32)
    root_xyzw[0] = row[1]
    root_xyzw[1] = row[0]
    root_xyzw[2] = row[2]
    root_rpy = np.asarray([row[4], row[3], -row[5]], dtype=np.float32)
    root_xyzw[3:7] = R.from_euler("xyz", root_rpy).as_quat().astype(np.float32)
    root_xyzw[7] = row[7]
    root_xyzw[8] = row[6]
    root_xyzw[9] = row[8]
    root_xyzw[10] = row[10]
    root_xyzw[11] = row[9]
    root_xyzw[12] = -row[11]
    return dof_pos, root_xyzw, root_rpy


def _quat_xyzw_to_wxyz_np(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return np.concatenate((quat[..., 3:4], quat[..., 0:3]), axis=-1)


def _quat_wxyz_to_xyzw_np(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return np.concatenate((quat[..., 1:4], quat[..., 0:1]), axis=-1)


def _quat_conjugate_xyzw(quat):
    out = quat.copy()
    out[..., :3] *= -1.0
    return out


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


def _finite_diff_pos(pos_buffer, data_dt):
    vel = np.zeros_like(pos_buffer)
    if len(pos_buffer) > 1:
        vel[1:] = (pos_buffer[1:] - pos_buffer[:-1]) / data_dt
    return vel


def _smooth_binary_contact(contact, min_len=2):
    contact = np.asarray(contact, dtype=np.float32).copy()
    if min_len <= 1 or contact.ndim != 2:
        return contact

    T, N = contact.shape
    for j in range(N):
        x = contact[:, j]
        start = 0
        while start < T:
            end = start + 1
            while end < T and x[end] == x[start]:
                end += 1

            if end - start < min_len:
                left = x[start - 1] if start > 0 else x[start]
                right = x[end] if end < T else x[start]
                if left == right:
                    x[start:end] = left

            start = end
        contact[:, j] = x
    return contact


def _finite_diff_quat_ang_vel(quat_buffer, data_dt):
    ang_vel = np.zeros(quat_buffer.shape[:-1] + (3,), dtype=quat_buffer.dtype)
    if len(quat_buffer) <= 1:
        return ang_vel
    q_prev = quat_buffer[:-1]
    q_curr = quat_buffer[1:]
    q_prev = q_prev / np.maximum(np.linalg.norm(q_prev, axis=-1, keepdims=True), 1e-8)
    q_curr = q_curr / np.maximum(np.linalg.norm(q_curr, axis=-1, keepdims=True), 1e-8)
    same_hemi = np.sum(q_prev * q_curr, axis=-1, keepdims=True) >= 0.0
    q_curr = np.where(same_hemi, q_curr, -q_curr)
    q_delta = _quat_multiply_xyzw(q_curr, _quat_conjugate_xyzw(q_prev))
    q_delta = q_delta / np.maximum(np.linalg.norm(q_delta, axis=-1, keepdims=True), 1e-8)
    rot_vec = q_delta[..., :3]
    rot_w = np.clip(q_delta[..., 3], -1.0, 1.0)
    rot_norm = np.linalg.norm(rot_vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(rot_norm, rot_w[..., None])
    axis = np.divide(rot_vec, rot_norm, out=np.zeros_like(rot_vec), where=rot_norm > 1e-8)
    ang_vel[1:] = axis * angle / data_dt
    return ang_vel


def _quat_rotate_inverse_xyzw(q, v):
    return R.from_quat(q).inv().apply(v).astype(np.float32)


def _recording_placeholder_motion_path():
    """Create a tiny valid trajectory file so the recording env can boot.

    The recorder only needs IsaacLab to instantiate the robot and report body
    poses for externally written states.  It should not depend on the dance
    keypoint file that it is about to generate, so we feed the trajectory env a
    minimal inert reference during startup/reset.
    """
    path = os.path.join("/tmp", "mrobot_record_dance_placeholder_keypoint.npz")
    if os.path.exists(path):
        return path

    frames = 8

    def zeros(*shape):
        return np.zeros((frames,) + shape, dtype=np.float32)

    def identity_quat(parts):
        quat = zeros(parts, 4)
        quat[..., 3] = 1.0
        return quat

    root_states = zeros(13)
    root_states[:, 2] = 0.9
    root_states[:, 6] = 1.0
    np.savez_compressed(
        path,
        dof_pos=zeros(29),
        dof_vel=zeros(29),
        root_states=root_states,
        root_linvel=zeros(3),
        root_angvel=zeros(3),
        euler_xyz=zeros(3),
        foot_height=zeros(2),
        feet_contact=np.ones((frames, 2), dtype=np.float32),
        pelvis_pos=zeros(1, 3),
        pelvis_vel=zeros(1, 3),
        pelvis_quat=identity_quat(1),
        pelvis_ang_vel=zeros(1, 3),
        feet_pos=zeros(2, 3),
        feet_vel=zeros(2, 3),
        feet_quat=identity_quat(2),
        feet_ang_vel=zeros(2, 3),
        knee_pos=zeros(2, 3),
        knee_vel=zeros(2, 3),
        knee_quat=identity_quat(2),
        knee_ang_vel=zeros(2, 3),
        hip_pos=zeros(2, 3),
        hip_vel=zeros(2, 3),
        hip_quat=identity_quat(2),
        hip_ang_vel=zeros(2, 3),
        pelvic_yaw_pos=zeros(2, 3),
        pelvic_yaw_vel=zeros(2, 3),
        pelvic_yaw_quat=identity_quat(2),
        pelvic_yaw_ang_vel=zeros(2, 3),
        waist_pos=zeros(1, 3),
        waist_vel=zeros(1, 3),
        waist_quat=identity_quat(1),
        waist_ang_vel=zeros(1, 3),
    )
    return path


def _configure_env(args):
    from humanoid_gym_ex.envs.robots.mrobot.isaaclab_env import MrobotMimicDanceIsaacLabEnv, MrobotMimicDanceIsaacLabEnvCfg

    env_cfg = MrobotMimicDanceIsaacLabEnvCfg()
    env_cfg.seed = 123145
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.sim.dt = PHYSICS_DT
    env_cfg.sim.render_interval = CONTROL_DECIMATION
    env_cfg.decimation = CONTROL_DECIMATION
    if hasattr(env_cfg, "contact_sensor"):
        env_cfg.contact_sensor.update_period = CONTROL_DT
    env_cfg.disable_domain_randomization = True
    env_cfg.deterministic_reset = True
    env_cfg.profile_step_timings = False
    env_cfg.motion_files = [_recording_placeholder_motion_path()]
    env = MrobotMimicDanceIsaacLabEnv(env_cfg)
    env.reset()
    return env


def _init_buffers():
    buffers = OrderedDict()
    buffers["dof_pos"] = []
    buffers["dof_vel"] = []
    buffers["root_states"] = []
    buffers["root_linvel"] = []
    buffers["root_angvel"] = []
    buffers["euler_xyz"] = []
    buffers["foot_height"] = []
    for field_name, _ in KEY_BODY_FIELDS:
        buffers[f"{field_name}_pos"] = []
        buffers[f"{field_name}_vel"] = []
        buffers[f"{field_name}_quat"] = []
        buffers[f"{field_name}_ang_vel"] = []
    buffers["feet_contact"] = []
    return buffers


def _collect_for_file(
    env,
    data_pos,
    data_dt,
    render=False,
    data_stride=1,
    contact_height_threshold=0.095,
    contact_velocity_threshold=0.25,
    contact_smooth_min_len=2,
    contact_force_threshold=1.0,
):
    _ = contact_force_threshold  # Deprecated compatibility option; contact is generated from foot z/vz below.
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    default_joint_pos = env.robot.data.default_joint_pos[0:1].clone()
    default_joint_vel = env.robot.data.default_joint_vel[0:1].clone()
    buffers = _init_buffers()
    last_dof_pos = None
    last_root = None
    last_rpy = None
    data_stride = max(1, int(data_stride))

    sample_indices = range(0, data_pos.shape[0], data_stride)
    for sample_idx, src_idx in enumerate(tqdm(sample_indices, desc="frames", leave=False)):
        dof_pos_canon, root_xyzw, root_rpy = _data_row_to_canonical_state(data_pos[src_idx])
        if last_dof_pos is None:
            dof_vel_canon = np.zeros_like(dof_pos_canon)
            root_xyzw[7:13] = 0.0
        else:
            dof_vel_canon = (dof_pos_canon - last_dof_pos) / data_dt
            root_xyzw[7:10] = (root_xyzw[0:3] - last_root[0:3]) / data_dt
            rpy_delta = root_rpy - last_rpy
            rpy_delta = np.where(rpy_delta > np.pi, rpy_delta - 2.0 * np.pi, rpy_delta)
            rpy_delta = np.where(rpy_delta < -np.pi, rpy_delta + 2.0 * np.pi, rpy_delta)
            root_xyzw[10:13] = rpy_delta / data_dt

        last_dof_pos = dof_pos_canon.copy()
        last_root = root_xyzw.copy()
        last_rpy = root_rpy.copy()

        joint_pos = default_joint_pos.repeat(env.num_envs, 1)
        joint_vel = default_joint_vel.repeat(env.num_envs, 1)
        joint_pos[:, env.joint_sim_ids] = torch.as_tensor(dof_pos_canon, dtype=torch.float, device=env.device)
        joint_vel[:, env.joint_sim_ids] = torch.as_tensor(dof_vel_canon, dtype=torch.float, device=env.device)
        root_wxyz = torch.as_tensor(root_xyzw.copy(), dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        root_wxyz[:, 3:7] = torch.as_tensor(_quat_xyzw_to_wxyz_np(root_xyzw[3:7]), dtype=torch.float, device=env.device)

        env.robot.write_root_pose_to_sim(root_wxyz[:, :7], env_ids)
        env.robot.write_root_velocity_to_sim(root_wxyz[:, 7:13], env_ids)
        env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)

        if sample_idx == 0:
            continue

        buffers["dof_pos"].append(dof_pos_canon.copy())
        buffers["dof_vel"].append(dof_vel_canon.copy())
        buffers["root_states"].append(root_xyzw.copy())
        buffers["root_linvel"].append(_quat_rotate_inverse_xyzw(root_xyzw[3:7], root_xyzw[7:10]))
        buffers["root_angvel"].append(_quat_rotate_inverse_xyzw(root_xyzw[3:7], root_xyzw[10:13]))
        buffers["euler_xyz"].append(root_rpy.copy())

        rigid = env.robot.data.body_state_w
        buffers["foot_height"].append(rigid[0, env.feet_indices, 2].detach().cpu().numpy())
        for field_name, indices_attr in KEY_BODY_FIELDS:
            indices = getattr(env, indices_attr)
            state = rigid[0, indices].detach().cpu().numpy()
            buffers[f"{field_name}_pos"].append(state[:, 0:3])
            buffers[f"{field_name}_quat"].append(_quat_wxyz_to_xyzw_np(state[:, 3:7]))
            buffers[f"{field_name}_vel"].append(state[:, 7:10])
            buffers[f"{field_name}_ang_vel"].append(state[:, 10:13])

        if render:
            env.sim.render()

    for key in list(buffers.keys()):
        buffers[key] = np.asarray(buffers[key], dtype=np.float32)

    for field_name, _ in KEY_BODY_FIELDS:
        buffers[f"{field_name}_vel"] = _finite_diff_pos(buffers[f"{field_name}_pos"], data_dt)
        buffers[f"{field_name}_ang_vel"] = _finite_diff_quat_ang_vel(buffers[f"{field_name}_quat"], data_dt)
    if buffers["feet_pos"].size > 0:
        feet_z = buffers["feet_pos"][:, :, 2]
        feet_vz = buffers["feet_vel"][:, :, 2]
        feet_contact = (
            (feet_z < float(contact_height_threshold))
            & (np.abs(feet_vz) < float(contact_velocity_threshold))
        ).astype(np.float32)
        feet_contact = _smooth_binary_contact(feet_contact, min_len=int(contact_smooth_min_len))
    else:
        feet_contact = np.zeros((0, 2), dtype=np.float32)
    buffers["feet_contact"] = feet_contact
    if len(buffers["dof_vel"]) > 0:
        buffers["dof_vel"][0] = 0.0
        buffers["root_states"][0, 7:13] = 0.0
        buffers["root_linvel"][0] = 0.0
        buffers["root_angvel"][0] = 0.0
    contact_ratio = buffers["feet_contact"].mean(axis=0) if len(buffers["feet_contact"]) > 0 else np.zeros(2, dtype=np.float32)
    print(
        "[record_dance_keypoints_lab] feet_contact generated by height+velocity+smoothing: "
        f"height_th={contact_height_threshold}, "
        f"vel_th={contact_velocity_threshold}, "
        f"smooth_min_len={contact_smooth_min_len}, "
        f"contact_ratio={contact_ratio}",
        flush=True,
    )
    return buffers


def _repeat_frame(buffer, repeat_frames, first=True, zero_all=False, zero_slice=None, fill_value=None):
    if repeat_frames <= 0 or len(buffer) == 0:
        return buffer
    frame = buffer[:1] if first else buffer[-1:]
    block = np.repeat(frame, repeat_frames, axis=0)
    if fill_value is not None:
        block[...] = fill_value
    if zero_all:
        block[...] = 0.0
    if zero_slice is not None:
        block[(slice(None),) + zero_slice] = 0.0
    return np.concatenate((block, buffer), axis=0) if first else np.concatenate((buffer, block), axis=0)


def _apply_still_segments(ref_np, prepend_frames, append_frames):
    zero_all_keys = {
        "dof_vel",
        "root_linvel",
        "root_angvel",
        "pelvis_vel",
        "pelvis_ang_vel",
        "feet_vel",
        "feet_ang_vel",
        "ankle_vel",
        "ankle_ang_vel",
        "knee_vel",
        "knee_ang_vel",
        "hip_vel",
        "hip_ang_vel",
        "pelvic_yaw_vel",
        "pelvic_yaw_ang_vel",
        "waist_vel",
        "waist_ang_vel",
    }
    for key in list(ref_np.keys()):
        zero_all = key in zero_all_keys
        zero_slice = (slice(7, 13),) if key == "root_states" else None
        fill_value = 1.0 if key == "feet_contact" else None
        ref_np[key] = _repeat_frame(ref_np[key], prepend_frames, first=True, zero_all=zero_all, zero_slice=zero_slice, fill_value=fill_value)
        ref_np[key] = _repeat_frame(ref_np[key], append_frames, first=False, zero_all=zero_all, zero_slice=zero_slice, fill_value=fill_value)
    return ref_np


def _clean_initial_frames(ref_np):
    source = CLEAN_INITIAL_COPY_SOURCE_FRAME
    if ref_np["dof_pos"].shape[0] <= source:
        return ref_np
    for key, value in ref_np.items():
        clean_frames = min(source, value.shape[0])
        if clean_frames > 0:
            value[:clean_frames] = value[source]
        ref_np[key] = value
    return ref_np


def _field_headers(field_name, field_sample):
    arr = np.asarray(field_sample)
    if arr.ndim == 0:
        return [field_name]
    if arr.ndim == 1:
        return [f"{field_name}_{i}" for i in range(arr.shape[0])]
    last_dim_names = ["x", "y", "z"] if arr.shape[-1] == 3 else ["x", "y", "z", "w"] if arr.shape[-1] == 4 else [str(i) for i in range(arr.shape[-1])]
    headers = []
    for prefix_idx in np.ndindex(*arr.shape[:-1]):
        prefix = field_name + "".join([f"_{idx}" for idx in prefix_idx])
        headers.extend([f"{prefix}_{name}" for name in last_dim_names])
    return headers


def _write_csv(csv_path, ref_np):
    fields = list(ref_np.keys())
    headers = []
    for field in fields:
        headers.extend(_field_headers(field, np.asarray(ref_np[field][0])))
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for frame_idx in range(ref_np["dof_pos"].shape[0]):
            row = []
            for field in fields:
                row.extend(np.asarray(ref_np[field][frame_idx]).reshape(-1).tolist())
            writer.writerow(row)


def _output_base_name(input_path):
    name = os.path.basename(input_path)
    base = name[:-5] if name.endswith(".data") else os.path.splitext(name)[0]
    base = base.replace("100HZ", "50HZ").replace("100hz", "50hz")
    return base


def record_dance_keypoints(args):
    input_paths = [args.file] if args.file else [os.path.join(args.input_dir, name) for name in DEFAULT_FILES]
    print(
        "[record_dance_keypoints_lab] start: "
        f"input_paths={input_paths}, output_dir={args.output_dir}, "
        f"data_dt={args.data_dt}, data_stride={args.data_stride}",
        flush=True,
    )
    missing = [path for path in input_paths if not os.path.isfile(path)]
    if missing and not args.allow_missing:
        raise FileNotFoundError(f"Missing input .data files: {missing[:3]}")
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    env = _configure_env(args)
    try:
        for input_path in tqdm(input_paths, desc="dance files"):
            if not os.path.isfile(input_path):
                continue
            data = _read_data(input_path)
            ref_np = _collect_for_file(
                env,
                data,
                args.data_dt,
                render=args.render,
                data_stride=args.data_stride,
                contact_height_threshold=args.contact_height_threshold,
                contact_velocity_threshold=args.contact_velocity_threshold,
                contact_smooth_min_len=args.contact_smooth_min_len,
                contact_force_threshold=args.contact_force_threshold,
            )
            prepend_frames = int(round(args.prepend_stand_s / args.data_dt))
            append_frames = int(round(args.append_stand_s / args.data_dt))
            ref_np = _apply_still_segments(ref_np, prepend_frames, append_frames)
            ref_np = _clean_initial_frames(ref_np)
            print(
                "[record_dance_keypoints_lab] save shapes: "
                f"dof_pos={ref_np['dof_pos'].shape}, root={ref_np['root_states'].shape}, feet={ref_np['feet_pos'].shape}",
                flush=True,
            )
            if args.no_save:
                continue
            base = _output_base_name(input_path)
            npz_path = os.path.join(args.output_dir, f"{base}_keypoint.npz")
            csv_path = os.path.join(args.output_dir, f"{base}_keypoint.csv")
            np.savez_compressed(npz_path, **ref_np)
            _write_csv(csv_path, ref_np)
            print(f"[record_dance_keypoints_lab] saved: {npz_path}", flush=True)
            print(f"[record_dance_keypoints_lab] saved: {csv_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    args_cli = _parse_args()
    _install_cpu_governor_stderr_filter(enabled=not args_cli.show_cpu_governor_warning)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[record_dance_keypoints_lab] IsaacSim app launched; building recording env...", flush=True)
    try:
        record_dance_keypoints(args_cli)
    except BaseException:
        print("[record_dance_keypoints_lab] ERROR before IsaacSim close:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[record_dance_keypoints_lab] closing IsaacSim app...", flush=True)
        simulation_app.close()


# python humanoid_gym_ex/scripts/record_dance_keypoints_lab.py file /home/weil/HumanoidGym-Ex/ref_pos/BS_GuangHuiSuiYue_DongTai_100HZ.data \
