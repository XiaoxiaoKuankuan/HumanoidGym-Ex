"""MuJoCo sim2sim for the MRobot specified-trajectory dance task.

This is the trajectory-reference counterpart of ``sim2sim_mimic.py``.  It
loads a dance ``*_keypoint.npz`` or ``*_keypoint.csv`` file, builds the new 61-dim policy input
(``42`` proprio + ``19`` current goal), and drives the 29-DOF MuJoCo model with
12 policy-controlled leg joints while the remaining joints follow reference.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - runtime dependency
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

try:
    import mujoco.viewer as mujoco_passive_viewer
except ImportError:  # pragma: no cover - optional viewer path
    mujoco_passive_viewer = None

try:
    import mujoco_viewer
except ImportError as exc:  # pragma: no cover - runtime dependency
    mujoco_viewer = None
    if _MUJOCO_IMPORT_ERROR is None:
        _MUJOCO_IMPORT_ERROR = exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config import MrobotMimicDanceCfg
from humanoid_gym_ex.scripts.sim2sim_mimic import (
    compute_body_midpoint,
    compute_world_com,
    get_body_world_pos,
    get_foot_forces,
    get_obs,
    init_fourbar_params,
    load_policy,
    parallel_to_serial_pos_np,
    parallel_to_serial_vel_np,
    parallel_xml_to_policy_pos_np,
    parallel_xml_to_policy_vel_np,
    pd_control,
    policy_parallel_to_xml_tau_np,
    serial_tau_to_parallel_policy_tau_np,
    serial_to_parallel_pos_np,
    set_initial_joint_state,
    wrap_to_pi_np,
)
from humanoid_gym_ex.utils.mrobot_trajectory_reference import DEFAULT_DANCE_MOTION_FILES, resolve_motion_file


START_MOVING_STEP = 3000


def _resolve_path(path):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(LEGGED_GYM_ROOT_DIR, path)


def _quat_xyzw_to_euler(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    quat_xyzw = quat_xyzw / max(np.linalg.norm(quat_xyzw), 1e-8)
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return R.from_quat(quat_xyzw).as_euler("xyz")


def _load_motion_npz(path):
    path = resolve_motion_file(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dance reference motion not found: {path}")
    motion = dict(np.load(path, allow_pickle=True))
    required = ["dof_pos", "dof_vel", "waist_pos", "waist_quat", "waist_vel", "waist_ang_vel"]
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    if motion["dof_pos"].ndim != 2 or motion["dof_pos"].shape[1] != 29:
        raise ValueError(f"{path} dof_pos must be [T, 29], got {motion['dof_pos'].shape}")
    print(f"[sim2sim_dance] loaded motion: {path} frames={motion['dof_pos'].shape[0]}", flush=True)
    return {key: np.asarray(value, dtype=np.float32) for key, value in motion.items()}


def _parse_flat_field_columns(headers, field_name):
    prefix = f"{field_name}_"
    cols = []
    for idx, header in enumerate(headers):
        if header == field_name:
            cols.append((idx, ()))
        elif header.startswith(prefix):
            suffix = header[len(prefix):]
            parts = suffix.split("_")
            if parts[-1] in ("x", "y", "z", "w"):
                numeric = parts[:-1]
                axis = {"x": 0, "y": 1, "z": 2, "w": 3}[parts[-1]]
                index_tuple = tuple(int(item) for item in numeric) + (axis,)
            else:
                index_tuple = tuple(int(item) for item in parts)
            cols.append((idx, index_tuple))
    if not cols:
        return None
    max_rank = max(len(index_tuple) for _, index_tuple in cols)
    if max_rank == 0:
        return np.asarray([idx for idx, _ in cols], dtype=np.int64), ()
    shape = []
    for dim in range(max_rank):
        shape.append(max(index_tuple[dim] for _, index_tuple in cols if len(index_tuple) > dim) + 1)
    return cols, tuple(shape)


def _load_motion_csv(path):
    path = _resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dance reference motion not found: {path}")
    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    fields = [
        "dof_pos",
        "dof_vel",
        "root_states",
        "root_linvel",
        "root_angvel",
        "euler_xyz",
        "foot_height",
        "feet_contact",
        "ref_foot_contact",
        "pelvis_pos",
        "pelvis_vel",
        "pelvis_quat",
        "pelvis_ang_vel",
        "feet_pos",
        "feet_vel",
        "feet_quat",
        "feet_ang_vel",
        "ankle_pos",
        "ankle_vel",
        "ankle_quat",
        "ankle_ang_vel",
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
    ]
    motion = {}
    for field in fields:
        parsed = _parse_flat_field_columns(headers, field)
        if parsed is None:
            continue
        cols, shape = parsed
        if shape == ():
            col_indices = np.asarray(cols, dtype=np.int64)
            motion[field] = data[:, col_indices]
            continue
        arr = np.zeros((data.shape[0],) + shape, dtype=np.float32)
        for col_idx, index_tuple in cols:
            arr[(slice(None),) + index_tuple] = data[:, col_idx]
        motion[field] = arr

    required = ["dof_pos", "dof_vel", "waist_pos", "waist_quat", "waist_vel", "waist_ang_vel"]
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"{path} missing required CSV fields: {missing}")
    if motion["dof_pos"].ndim != 2 or motion["dof_pos"].shape[1] != 29:
        raise ValueError(f"{path} dof_pos must parse as [T, 29], got {motion['dof_pos'].shape}")
    print(f"[sim2sim_dance] loaded CSV motion: {path} frames={motion['dof_pos'].shape[0]}", flush=True)
    return motion


def _load_motion(path):
    path_str = str(path)
    if path_str.lower().endswith(".csv"):
        return _load_motion_csv(path_str)
    return _load_motion_npz(path_str)


def _build_goal(motion, idx, control_indices, obs_scales, zero_ref_motion=False):
    ref_dof = motion["dof_pos"][idx].astype(np.float64).copy()
    ref_dof_pos_curr = ref_dof[np.asarray(control_indices, dtype=np.int64)].astype(np.float32)
    waist_quat = motion["waist_quat"][idx, 0]
    waist_rp = _quat_xyzw_to_euler(waist_quat)[:2].astype(np.float32)
    waist_linvel = motion["waist_vel"][idx, 0].astype(np.float32) * float(obs_scales.lin_vel)
    waist_angvel_z = motion["waist_ang_vel"][idx, 0, 2:3].astype(np.float32) * float(obs_scales.ang_vel)
    if zero_ref_motion:
        waist_linvel[:] = 0.0
        waist_angvel_z[:] = 0.0
    goal = np.concatenate(
        [
            ref_dof_pos_curr,
            motion["waist_pos"][idx, 0, 2:3].astype(np.float32),
            waist_rp,
            waist_linvel,
            waist_angvel_z,
        ]
    ).astype(np.float32)
    return ref_dof, goal


def _read_terminal_key():
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    return sys.stdin.read(1)


def _create_mujoco_viewer(model, data, headless=False):
    if headless:
        return None, "headless"
    if mujoco_passive_viewer is not None:
        try:
            viewer = mujoco_passive_viewer.launch_passive(model, data)
            print("[sim2sim_dance] 使用官方 MuJoCo passive viewer，低频 sync 可加速可视化", flush=True)
            return viewer, "passive"
        except Exception as exc:
            print(f"[sim2sim_dance] 官方 passive viewer 启动失败，回退 mujoco_viewer: {exc}", flush=True)
    if mujoco_viewer is None:
        raise ImportError("mujoco_viewer fallback is not installed and passive viewer failed")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    print("[sim2sim_dance] 使用 mujoco_viewer 回退模式，仍按低频 render 刷新", flush=True)
    return viewer, "mujoco_viewer"


def _sync_mujoco_viewer(viewer, viewer_type):
    if viewer is None:
        return True
    if viewer_type == "passive":
        if not viewer.is_running():
            return False
        viewer.sync()
        return True
    viewer.render()
    return True


def _close_mujoco_viewer(viewer):
    if viewer is not None:
        viewer.close()


def run_mujoco(policy, cfg):
    if mujoco is None:
        raise ImportError("mujoco is required for sim2sim_dance") from _MUJOCO_IMPORT_ERROR
    if (
        not bool(getattr(cfg.sim_config, "headless", False))
        and mujoco_passive_viewer is None
        and mujoco_viewer is None
    ):
        raise ImportError("mujoco.viewer or mujoco_viewer is required for visual sim2sim_dance") from _MUJOCO_IMPORT_ERROR

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    viewer, viewer_type = _create_mujoco_viewer(
        model,
        data,
        headless=bool(getattr(cfg.sim_config, "headless", False)),
    )
    mujoco.mj_forward(model, data)
    decimation = int(cfg.sim_config.decimation)
    viewer_render_interval = max(1, int(getattr(cfg.sim_config, "viewer_render_interval", 20)))
    playback_speed = float(
        getattr(
            cfg.sim_config,
            "playback_speed",
            1.0 if bool(getattr(cfg.sim_config, "real_time", False)) else 0.0,
        )
    )
    physics_hz = 1.0 / float(cfg.sim_config.dt)
    control_hz = physics_hz / decimation
    viewer_hz = physics_hz / viewer_render_interval
    print(
        "[sim2sim_dance] 主循环配置: "
        f"physics={physics_hz:.0f}Hz, policy/control={control_hz:.0f}Hz, "
        f"viewer 每 {viewer_render_interval} 个底层 step 刷新一次(约 {viewer_hz:.0f}Hz), "
        f"playback_speed={'不限速' if playback_speed <= 0 else playback_speed}",
        flush=True,
    )

    keyboard_fd = None
    keyboard_old_termios = None
    if sys.stdin.isatty():
        keyboard_fd = sys.stdin.fileno()
        keyboard_old_termios = termios.tcgetattr(keyboard_fd)
        tty.setcbreak(keyboard_fd)
        print("[sim2sim_dance] 终端键盘控制：S 切换第一帧保持/恢复", flush=True)

    def _restore_keyboard():
        if keyboard_old_termios is not None and keyboard_fd is not None:
            termios.tcsetattr(keyboard_fd, termios.TCSADRAIN, keyboard_old_termios)

    atexit.register(_restore_keyboard)

    foot_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_l6_link"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_r6_link"),
    ]
    foot_body_ids = [body_id for body_id in foot_body_ids if body_id >= 0]
    left_foot_body_id = foot_body_ids[0] if len(foot_body_ids) > 0 else -1
    right_foot_body_id = foot_body_ids[1] if len(foot_body_ids) > 1 else -1
    waist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, getattr(cfg.asset, "waist_name", "waist_yaw_link"))

    motion = _load_motion(cfg.sim_config.motion_file)
    motion_length = int(motion["dof_pos"].shape[0])
    control_idx = np.asarray(cfg.env.num_control, dtype=np.int64)
    n_ctrl = len(control_idx)

    fourbar = init_fourbar_params()
    initial_ref_dof = motion["dof_pos"][0].astype(np.float64).copy()
    initial_parallel = serial_to_parallel_pos_np(initial_ref_dof, fourbar)
    set_initial_joint_state(model, data, initial_ref_dof, initial_parallel)
    mujoco.mj_forward(model, data)
    print("[sim2sim_dance] 已按 reference 第一帧初始化 MuJoCo 关节角", flush=True)

    action = np.zeros(cfg.env.num_actions, dtype=np.float64)
    last_action = np.zeros_like(action)
    target_q_filter = np.zeros_like(action)
    delayed_target_q_filter = np.zeros_like(action)
    action_delay_buffer = np.zeros((cfg.env.num_actions, cfg.sim_config.action_delay + 1), dtype=np.float64)

    ref_dof_val, goal_buf = _build_goal(motion, 0, control_idx, cfg.normalization.obs_scales, zero_ref_motion=True)
    hold_ref_dof_val = ref_dof_val.copy()
    hold_goal_buf = goal_buf.copy()
    hold_first_frame = False
    start_dance = False
    initial_base_yaw = None
    ref_idx = 0

    tau_buffer, q_buffer, dq_buffer, ref_buffer = [], [], [], []
    final_target_buffer, action_buffer, foot_force_buffer = [], [], []

    for count_lowlevel in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="sim2sim_dance"):
        step_start = time.time()
        if keyboard_fd is not None:
            key = _read_terminal_key()
            if key in ("s", "S"):
                hold_first_frame = not hold_first_frame
                print(f"[sim2sim_dance] hold_first_frame={hold_first_frame}", flush=True)

        _, _, quat, _, omega, _ = get_obs(data)
        q_parallel = parallel_xml_to_policy_pos_np(np.asarray(data.actuator_length, dtype=np.float64))
        dq_parallel = parallel_xml_to_policy_vel_np(np.asarray(data.actuator_velocity, dtype=np.float64))
        q = parallel_to_serial_pos_np(q_parallel, fourbar)
        dq = parallel_to_serial_vel_np(q, q_parallel, dq_parallel, fourbar)
        obs_euler = R.from_quat(quat).as_euler("xyz")
        if start_dance and initial_base_yaw is not None:
            obs_euler[2] = wrap_to_pi_np(obs_euler[2] - initial_base_yaw)
        else:
            obs_euler[2] = 0.0

        if (not start_dance) and cfg.sim_config.static_com_log_interval > 0:
            if count_lowlevel % cfg.sim_config.static_com_log_interval == 0:
                com_world = compute_world_com(model, data)
                feet_mid = compute_body_midpoint(data, foot_body_ids)
                waist_world = get_body_world_pos(data, waist_body_id)
                msg = f"[sim2sim_dance][static] step={count_lowlevel} COM=({com_world[0]:.4f},{com_world[1]:.4f},{com_world[2]:.4f})"
                if feet_mid is not None:
                    msg += f" feet_mid=({feet_mid[0]:.4f},{feet_mid[1]:.4f},{feet_mid[2]:.4f})"
                if waist_world is not None:
                    msg += f" waist=({waist_world[0]:.4f},{waist_world[1]:.4f},{waist_world[2]:.4f})"
                print(msg, flush=True)

        if count_lowlevel % decimation == 0:
            last_action[:] = action
            if start_dance and not hold_first_frame:
                ref_idx = min(ref_idx + 1, motion_length - 1)
            else:
                ref_idx = 0
            zero_ref_motion = (not start_dance) or hold_first_frame
            ref_dof_val, goal_buf = _build_goal(
                motion,
                ref_idx,
                control_idx,
                cfg.normalization.obs_scales,
                zero_ref_motion=zero_ref_motion,
            )
            if hold_first_frame:
                ref_dof_val = hold_ref_dof_val.copy()
                goal_buf = hold_goal_buf.copy()

            obs = np.zeros((1, cfg.env.num_single_obs), dtype=np.float32)
            obs[0, 0:n_ctrl] = (q[control_idx] - ref_dof_val[control_idx]) * cfg.normalization.obs_scales.dof_pos
            obs[0, n_ctrl : 2 * n_ctrl] = dq[control_idx] * cfg.normalization.obs_scales.dof_vel
            obs[0, 2 * n_ctrl : 3 * n_ctrl] = action[control_idx]
            offset = 3 * n_ctrl
            obs[0, offset : offset + 3] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, offset + 3 : offset + 6] = obs_euler * cfg.normalization.obs_scales.quat

            policy_input = np.concatenate((obs.reshape(-1), goal_buf)).reshape(1, -1).astype(np.float32)
            if policy_input.shape[1] != cfg.env.num_observations:
                raise RuntimeError(
                    f"policy input dim mismatch: got {policy_input.shape[1]}, expected {cfg.env.num_observations}"
                )
            with torch.no_grad():
                rl_out = policy(torch.from_numpy(policy_input))[0].numpy()
            if rl_out.shape[0] != n_ctrl:
                raise RuntimeError(f"policy output dim mismatch: got {rl_out.shape[0]}, expected {n_ctrl}")

            raw_action = np.zeros(cfg.env.num_actions, dtype=np.float64)
            raw_action[control_idx] = rl_out
            raw_action[np.asarray(cfg.env.num_notcontrol, dtype=np.int64)] = (
                ref_dof_val[np.asarray(cfg.env.ref_num_notcontrol, dtype=np.int64)] / cfg.control.action_scale
            )
            action[:] = np.clip(raw_action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

        if cfg.normalization.actions_filter:
            rate = (count_lowlevel % decimation + 1.0) / decimation
            action_filter = (1.0 - rate) * last_action + rate * action
            target_q_filter[:] = action_filter * cfg.control.action_scale
        else:
            target_q_filter[:] = action * cfg.control.action_scale

        if cfg.sim_config.action_delay > 0:
            action_delay_buffer[:, 1:] = action_delay_buffer[:, :-1]
            action_delay_buffer[:, 0] = target_q_filter
            delayed_target_q_filter[:] = action_delay_buffer[:, cfg.sim_config.action_delay]
        else:
            delayed_target_q_filter[:] = target_q_filter

        final_target = ref_dof_val.copy()
        if getattr(cfg.control, "use_ref_residual_target", True):
            final_target[control_idx] = ref_dof_val[control_idx] + delayed_target_q_filter[control_idx]
        else:
            default_dof = motion["dof_pos"][0].astype(np.float64)
            final_target[control_idx] = default_dof[control_idx] + delayed_target_q_filter[control_idx]

        tau_serial = pd_control(final_target, q, cfg.robot_config.kps, np.zeros_like(dq), dq, cfg.robot_config.kds)
        tau_serial = np.clip(tau_serial, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        tau_parallel = serial_tau_to_parallel_policy_tau_np(tau_serial, q, q_parallel, fourbar)
        tau_xml = policy_parallel_to_xml_tau_np(tau_parallel)
        data.ctrl[:] = tau_xml

        tau_buffer.append(tau_serial.copy())
        q_buffer.append(q.copy())
        dq_buffer.append(dq.copy())
        ref_buffer.append(ref_dof_val.copy())
        final_target_buffer.append(final_target.copy())
        action_buffer.append(action.copy())
        lf, rf, _ = get_foot_forces(model, data, left_foot_body_id, right_foot_body_id)
        foot_force_buffer.append(np.concatenate((lf, rf)))

        mujoco.mj_step(model, data)
        if count_lowlevel % viewer_render_interval == 0:
            if not _sync_mujoco_viewer(viewer, viewer_type):
                print("[sim2sim_dance] viewer 已关闭，结束仿真", flush=True)
                break

        if (not start_dance) and count_lowlevel >= START_MOVING_STEP:
            start_dance = True
            ref_idx = 0
            initial_base_yaw = R.from_quat(quat).as_euler("xyz")[2]
            print(f"--- Start Dance Reference at low-level step {START_MOVING_STEP} ---", flush=True)

        if playback_speed > 0.0:
            elapsed = time.time() - step_start
            sleep_time = cfg.sim_config.dt / playback_speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    _close_mujoco_viewer(viewer)
    print(
        "[sim2sim_dance] finished. "
        f"q={np.asarray(q_buffer).shape}, ref={np.asarray(ref_buffer).shape}, "
        f"action={np.asarray(action_buffer).shape}, foot_force={np.asarray(foot_force_buffer).shape}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MRobot dance sim2sim in MuJoCo.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to .pt JIT or .onnx policy.")
    parser.add_argument("--motion_file", type=str, default=DEFAULT_DANCE_MOTION_FILES[1], help="Dance *_keypoint.npz or *_keypoint.csv path.")
    parser.add_argument("--terrain", action="store_true", help="Use terrain MuJoCo xml instead of plane.")
    parser.add_argument("--duration", type=float, default=195.0)
    parser.add_argument("--real_time", action="store_true", help="Throttle MuJoCo loop to real time.")
    parser.add_argument("--headless", action="store_true", help="Run MuJoCo without viewer.")
    parser.add_argument(
        "--viewer_render_interval",
        type=int,
        default=20,
        help="Render/sync viewer every N low-level 1000Hz steps. 10=100Hz, 20=50Hz.",
    )
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=None,
        help="Wall-clock throttle speed. 0 means unlimited; 1 means real time; 2 means 2x real time.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Sim2simDanceCfg(MrobotMimicDanceCfg):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/Mrobot/mjcf/mjmodel_terrain.xml"
            else:
                mujoco_model_path = (
                    f"{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/"
                    "Serial/xml/CASBOT_02_shell_ENCOS_7dof_par_bass.xml"
                )
            motion_file = args.motion_file
            sim_duration = args.duration
            dt = 0.001
            decimation = 10
            action_delay = 0
            static_com_log_interval = 1000
            real_time = args.real_time
            headless = args.headless
            viewer_render_interval = max(1, int(args.viewer_render_interval))
            if args.playback_speed is None:
                playback_speed = 1.0 if args.real_time else 0.0
            else:
                playback_speed = float(args.playback_speed)

        class robot_config:
            kps = np.array(
                [
                    276.348923229 / 2,
                    276.348923229 / 2,
                    256.6097056 / 2,
                    276.348923229 / 2,
                    153.965828656 / 2,
                    153.965828656 / 2,
                    276.348923229 / 2,
                    276.348923229 / 2,
                    256.6097056 / 2,
                    276.348923229 / 2,
                    153.965828656 / 2,
                    153.965828656 / 2,
                    153.965828656 / 2,
                    *([200.0] * 16),
                ],
                dtype=np.float64,
            )
            kds = np.array(
                [
                    17.5929188596 / 2,
                    17.5929188596 / 2,
                    16.33628152 / 2,
                    17.5929188596 / 2,
                    9.80176907892 / 2,
                    9.80176907892 / 2,
                    17.5929188596 / 2,
                    17.5929188596 / 2,
                    16.33628152 / 2,
                    17.5929188596 / 2,
                    9.80176907892 / 2,
                    9.80176907892 / 2,
                    9.80176907892 / 2,
                    *([5.0] * 16),
                ],
                dtype=np.float64,
            )
            tau_limit = np.array(
                [
                    66.7,
                    86.7,
                    60.1,
                    86.7,
                    31.5,
                    31.5,
                    66.7,
                    86.7,
                    60.1,
                    86.7,
                    31.5,
                    31.5,
                    *([35.2] * 17),
                ],
                dtype=np.float64,
            )

    policy = load_policy(args.load_model)
    run_mujoco(policy, Sim2simDanceCfg())
