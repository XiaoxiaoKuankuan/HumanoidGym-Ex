import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
try:
    import mujoco
    import mujoco_viewer
    import glfw

    _MUJOCO_IMPORT_ERROR = None
except ImportError as exc:
    mujoco = None
    mujoco_viewer = None
    glfw = None
    _MUJOCO_IMPORT_ERROR = exc
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config import MrobotMimicCfg
from humanoid_gym_ex.scripts.space4bar import Space4Bar
from humanoid_gym_ex.utils.reference_state import ReferenceStateNet
import torch
import pandas as pd 
import matplotlib.pyplot as plt
import time
import csv
import select
import termios
import tty
import atexit
try:
    import pygame
except ImportError:
    pygame = None
from threading import Thread

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ====================== ONNX 策略封装（与 JIT 相同调用方式） ======================

class OnnxPolicyWrapper:
    """封装 ONNX 推理，使调用方式与 torch.jit 一致：policy(tensor)[0].numpy()。若 ONNX 已含归一化，输入为 raw obs。"""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor):
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        out = self.session.run(None, {self.input_name: obs_np})
        result = np.asarray(out[self.output_index], dtype=np.float32)
        return torch.from_numpy(result)


def load_policy(path):
    """根据扩展名加载 JIT (.pt) 或 ONNX (.onnx) 策略，返回统一可调用对象。"""
    path = str(path)
    if path.lower().endswith(".onnx"):
        if ort is None:
            raise ImportError("onnxruntime 未安装，无法加载 ONNX 模型。请: pip install onnxruntime")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        print(f"已加载 ONNX 策略: {path} (输入: {input_name})")
        return OnnxPolicyWrapper(session, input_name)
    else:
        print(f"已加载 JIT 策略: {path}")
        return torch.jit.load(path)


def resolve_repo_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


# ====================== 参考动作生成网络 ======================

REFERENCE_DOF_POS_COLUMNS = [
    "left_leg_pelvic_pitch_pos",
    "left_leg_pelvic_roll_pos",
    "left_leg_pelvic_yaw_pos",
    "left_leg_knee_pitch_pos",
    "left_leg_ankle_pitch_pos",
    "left_leg_ankle_roll_pos",
    "right_leg_pelvic_pitch_pos",
    "right_leg_pelvic_roll_pos",
    "right_leg_pelvic_yaw_pos",
    "right_leg_knee_pitch_pos",
    "right_leg_ankle_pitch_pos",
    "right_leg_ankle_roll_pos",
    "waist_yaw_pos",
    "left_shoulder_pitch_pos",
    "left_shoulder_roll_pos",
    "left_shoulder_yaw_pos",
    "left_elbow_pitch_pos",
    "left_wrist_yaw_pos",
    "left_wrist_pitch_pos",
    "left_wrist_roll_pos",
    "right_shoulder_pitch_pos",
    "right_shoulder_roll_pos",
    "right_shoulder_yaw_pos",
    "right_elbow_pitch_pos",
    "right_wrist_yaw_pos",
    "right_wrist_pitch_pos",
    "right_wrist_roll_pos",
]


class ReferenceMotionGenerator:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        model_path = resolve_repo_path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reference model checkpoint not found: {model_path}. "
                "Use --reference_model to pass BPM_dance/reference_state_keypoint_model.pt or an absolute checkpoint path."
            )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.output_columns = list(checkpoint["output_columns"])
        self.column_index = {name: idx for idx, name in enumerate(self.output_columns)}
        self.bpm_mean = float(checkpoint["bpm_mean"])
        self.bpm_std = max(float(checkpoint["bpm_std"]), 1e-6)
        self.target_mean = torch.as_tensor(checkpoint["target_mean"], device=self.device, dtype=torch.float32)
        self.target_std = torch.as_tensor(checkpoint["target_std"], device=self.device, dtype=torch.float32)
        print(f"ReferenceMotionGenerator: BPM 归一化 (mean={self.bpm_mean:.2f}, std={self.bpm_std:.2f})")
        self.model = ReferenceStateNet(
            int(checkpoint["input_dim"]),
            int(checkpoint["output_dim"]),
            checkpoint["hidden"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def normalized_bpm(self, bpm):
        return (float(bpm) - self.bpm_mean) / self.bpm_std

    def predict(self, bpm, phase_rad):
        encoded = torch.tensor(
            [[self.normalized_bpm(bpm), math.sin(phase_rad), math.cos(phase_rad)]],
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred_norm = self.model(encoded)
            pred = pred_norm * self.target_std + self.target_mean
        return pred[0].cpu().numpy().astype(np.float64)

    def get(self, pred, name, default=0.0):
        idx = self.column_index.get(name)
        if idx is None:
            return float(default)
        return float(pred[idx])

    def build_reference(self, bpm, phase_rad, default_dof_pos, controlled_indices=None, zero_ref_motion=False):
        pred = self.predict(bpm, phase_rad)
        ref_dof = default_dof_pos.copy()
        for idx, column in enumerate(REFERENCE_DOF_POS_COLUMNS):
            if idx >= len(ref_dof):
                break
            ref_dof[idx] = self.get(pred, column, ref_dof[idx])

        if controlled_indices is None:
            controlled_indices = list(range(12))
        ref_dof_pos_curr = ref_dof[np.asarray(controlled_indices, dtype=np.int64)].astype(np.float32)
        ref_waist_quat_xyzw = standardize_quat(
            np.array(
                [
                    self.get(pred, "waist_quat_0_x", 0.0),
                    self.get(pred, "waist_quat_0_y", 0.0),
                    self.get(pred, "waist_quat_0_z", 0.0),
                    self.get(pred, "waist_quat_0_w", 1.0),
                ],
                dtype=np.float64,
            )
        )
        ref_waist_rp = R.from_quat(ref_waist_quat_xyzw).as_euler("xyz")[:2].astype(np.float32)
        ref_waist_pos_z = np.array([self.get(pred, "waist_pos_0_z", 0.0)], dtype=np.float32)
        ref_waist_linvel = np.array(
            [
                self.get(pred, "waist_vel_0_x", 0.0),
                self.get(pred, "waist_vel_0_y", 0.0),
                self.get(pred, "waist_vel_0_z", 0.0),
            ],
            dtype=np.float32,
        )
        ref_waist_angvel_z = np.array([self.get(pred, "waist_ang_vel_0_z", 0.0)], dtype=np.float32)
        if zero_ref_motion:
            ref_waist_linvel[:] = 0.0
            ref_waist_angvel_z[:] = 0.0

        goal_buf = np.concatenate(
            [
                ref_dof_pos_curr,
                ref_waist_pos_z,
                ref_waist_rp,
                ref_waist_linvel,
                ref_waist_angvel_z,
            ]
        ).astype(np.float32)
        return ref_dof, goal_buf


# ====================== 辅助函数 (保持不变) ======================

def standardize_quat(q):
    """
    强制四元数的 w 分量为正。
    兼容形状 (4,) 和 (N, 4)。
    """
    q = np.array(q, dtype=np.float32)
    if q.ndim == 2:
        neg_indices = q[:, -1] < 0
        q[neg_indices] *= -1
    elif q.ndim == 1:
        if q[-1] < 0:
            q *= -1
    return q

def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]], dtype=q.dtype)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)

def load_ref_data_csv(csv_path):
    print(f"Loading reference motion from: {csv_path}")
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)

    num_frames = data.shape[0]
    total_cols = data.shape[1]
    if total_cols < 71:
        print(f"[Warning] 参考 CSV 仅有 {total_cols} 列，请确认文件是否完整。")

    buffer = {
        "raw_data": data,
        "length": num_frames,
        "columns": list(df.columns),
    }
    return buffer


def _build_ref_column_index(columns):
    return {name: idx for idx, name in enumerate(columns)}


def _get_ref_values(row, column_index, names):
    return np.asarray([row[column_index[name]] for name in names], dtype=np.float32)

def _setup_matplotlib_chinese_font():
    """修复中文标题/图例显示为方框：优先选用系统里确实存在的中文字体。"""
    from matplotlib import font_manager

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = []
    for name in candidates:
        if name in available:
            chosen.append(name)
    if not chosen:
        print(
            "[sim2sim] 未检测到常见中文字体，中文可能仍显示为方框；"
            "可安装: sudo apt install fonts-noto-cjk"
        )
    base = list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["font.sans-serif"] = chosen + base
    plt.rcParams["axes.unicode_minus"] = False


def get_obs(data):
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) # IMU Quat (x,y,z,w)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def wrap_to_pi_np(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


SERIAL_DOF_JOINT_NAMES = [
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "waist_yaw_joint",
    "upper_left_1_joint",
    "upper_left_2_joint",
    "upper_left_3_joint",
    "upper_left_4_joint",
    "upper_left_5_joint",
    "upper_left_6_joint",
    "upper_left_7_joint",
    "upper_right_1_joint",
    "upper_right_2_joint",
    "upper_right_3_joint",
    "upper_right_4_joint",
    "upper_right_5_joint",
    "upper_right_6_joint",
    "upper_right_7_joint",
    "vhead_1_joint",
    "vhead_2_joint",
]

PARALLEL_MOTOR_JOINT_NAMES = {
    4: "lleg_4_2",
    5: "lleg_4_1",
    10: "rleg_4_2",
    11: "rleg_4_1",
}


def set_joint_qpos_by_name(model, data, joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        print(f"[sim2sim_mimic] Warning: joint '{joint_name}' not found in MuJoCo model")
        return False
    data.qpos[int(model.jnt_qposadr[joint_id])] = value
    return True


def set_joint_qvel_by_name(model, data, joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return False
    data.qvel[int(model.jnt_dofadr[joint_id])] = value
    return True


def set_initial_joint_state(model, data, dof_pos_serial, dof_pos_parallel):
    # MuJoCo qpos 顺序是 XML joint 顺序，不是 actuator 顺序。
    # 踝部尤其容易错：qpos 中有被动串联 leg_l5/leg_l6，actuator 中是四连杆电机 lleg/rleg_4_*。
    # 因此必须按 joint 名写入，不能把 29 维 action/actuator 向量直接切片塞进 qpos。
    for idx, joint_name in enumerate(SERIAL_DOF_JOINT_NAMES):
        if idx >= len(dof_pos_serial):
            break
        set_joint_qpos_by_name(model, data, joint_name, float(dof_pos_serial[idx]))
        set_joint_qvel_by_name(model, data, joint_name, 0.0)

    for idx, joint_name in PARALLEL_MOTOR_JOINT_NAMES.items():
        if idx >= len(dof_pos_parallel):
            continue
        set_joint_qpos_by_name(model, data, joint_name, float(dof_pos_parallel[idx]))
        set_joint_qvel_by_name(model, data, joint_name, 0.0)


# ====================== Space4Bar 串并联踝关节解算入口 ======================
#
# 保留 sim2sim 主流程里原来的函数名，但实际解算全部委托给
# humanoid/scripts/space4bar.py，避免这里维护第二套四连杆公式。

def init_fourbar_params():
    return Space4Bar()


def serial_to_parallel_pos_np(dof_pos_serial, fourbar):
    dof_pos_parallel = np.array(dof_pos_serial, dtype=np.float64, copy=True)
    dof_pos_parallel[4:6] = fourbar.left4BarIK(dof_pos_parallel[4:6])
    dof_pos_parallel[10:12] = fourbar.right4BarIK(dof_pos_parallel[10:12])
    return dof_pos_parallel


def parallel_to_serial_pos_np(dof_pos_parallel, fourbar):
    dof_pos_serial = np.array(dof_pos_parallel, dtype=np.float64, copy=True)
    dof_pos_serial[4:6] = fourbar.leftFkBase(dof_pos_parallel[4:6])
    dof_pos_serial[10:12] = fourbar.rightFkBase(dof_pos_parallel[10:12])
    return dof_pos_serial


def parallel_xml_to_policy_pos_np(dof_pos_xml):
    return np.array(dof_pos_xml, dtype=np.float64, copy=True)


def parallel_xml_to_policy_vel_np(dof_vel_xml):
    return np.array(dof_vel_xml, dtype=np.float64, copy=True)


def policy_parallel_to_xml_pos_np(dof_pos_policy):
    return np.array(dof_pos_policy, dtype=np.float64, copy=True)


def policy_parallel_to_xml_tau_np(tau_policy):
    return np.array(tau_policy, dtype=np.float64, copy=True)


def parallel_to_serial_vel_np(dof_pos_serial, dof_pos_parallel, dof_vel_parallel, fourbar):
    dof_vel_serial = np.array(dof_vel_parallel, dtype=np.float64, copy=True)
    dof_vel_serial[4:6] = fourbar.leftP2SVel(dof_pos_parallel[4:6], dof_vel_parallel[4:6])
    dof_vel_serial[10:12] = fourbar.rightP2SVel(dof_pos_parallel[10:12], dof_vel_parallel[10:12])
    return dof_vel_serial


def serial_tau_to_parallel_policy_tau_np(tau_serial, q_serial, q_parallel, fourbar):
    tau_parallel = np.array(tau_serial, dtype=np.float64, copy=True)
    tau_parallel[4:6] = fourbar.leftS2PTorque(q_serial[4:6], tau_serial[4:6])
    tau_parallel[10:12] = fourbar.rightS2PTorque(q_serial[10:12], tau_serial[10:12])
    return tau_parallel


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands'''
    return (target_q - q) * kp + (target_dq - dq) * kd

def find_body_ids_by_substring(model, substring):
    body_ids = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is not None and substring in body_name:
            body_ids.append(body_id)
    return body_ids

def compute_world_com(model, data):
    body_ids = np.arange(1, model.nbody)
    body_masses = model.body_mass[body_ids][:, None]
    total_mass = np.sum(body_masses)
    if total_mass <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.sum(body_masses * data.xipos[body_ids], axis=0) / total_mass

def compute_body_midpoint(data, body_ids):
    if len(body_ids) == 0:
        return None
    return np.mean(np.asarray(data.xpos[body_ids]), axis=0)

def get_body_world_pos(data, body_id):
    if body_id is None or body_id < 0:
        return None
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def get_body_world_quat_xyzw(data, body_id):
    if body_id is None or body_id < 0:
        return None
    return quat_wxyz_to_xyzw(np.asarray(data.xquat[body_id], dtype=np.float64))


def foot_forces_from_contacts(model, data, left_body_id, right_body_id):
    """无足底力传感器时：由各接触点的 mj_contactForce 变换到世界系后按左右足累加。"""
    fl = np.zeros(3, dtype=np.float64)
    fr = np.zeros(3, dtype=np.float64)
    if left_body_id < 0 and right_body_id < 0:
        return fl, fr
    for i in range(data.ncon):
        g1 = int(data.contact.geom1[i])
        g2 = int(data.contact.geom2[i])
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        f6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, f6)
        R = np.asarray(data.contact[i].frame, dtype=np.float64).reshape(3, 3)
        f_w = R @ f6[:3]
        if left_body_id >= 0 and (b1 == left_body_id or b2 == left_body_id):
            fl += f_w
        if right_body_id >= 0 and (b1 == right_body_id or b2 == right_body_id):
            fr += f_w
    return fl, fr


def get_foot_forces(model, data, left_body_id, right_body_id):
    """
    返回左右足底力 (3,) 与来源标记。
    优先使用 MJCF 中 <force site='left_force/right_force'> 传感器（site 坐标系三分量）；
    若无传感器（如部分 terrain 模型），则用接触力累加（世界系）。
    """
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_force")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_force")
    if lid >= 0 and rid >= 0:
        la, ld = model.sensor_adr[lid], model.sensor_dim[lid]
        ra, rd = model.sensor_adr[rid], model.sensor_dim[rid]
        lf = np.array(data.sensordata[la : la + ld], dtype=np.float64).ravel()[:3]
        rf = np.array(data.sensordata[ra : ra + rd], dtype=np.float64).ravel()[:3]
        return lf, rf, "sensor_site"
    lf, rf = foot_forces_from_contacts(model, data, left_body_id, right_body_id)
    return lf, rf, "contact_world"

# ====================== 主运行逻辑 (已修改) ======================

BPM_MIN = 60.0
BPM_MAX = 170.0
BPM_STEP = 5.0
DEFAULT_BPM = 70.0
PRESTART_BPM = 0.0
START_MOVING_STEP = 3000
BPM_LOG_INTERVAL_SEC = 1.0


def _read_terminal_key(stdin):
    readable, _, _ = select.select([stdin], [], [], 0.0)
    if not readable:
        return None

    chars = [stdin.read(1)]
    # 方向键在终端里通常是 escape 序列：上键 "\x1b[A"，下键 "\x1b[B"。
    # cbreak 模式下三个字符不一定同一瞬间都到，因此读到 ESC 后稍等一下把剩余字符补齐。
    if chars[0] == "\x1b":
        deadline = time.time() + 0.02
        while time.time() < deadline and len(chars) < 3:
            readable, _, _ = select.select([stdin], [], [], max(0.0, deadline - time.time()))
            if not readable:
                break
            chars.append(stdin.read(1))

    while True:
        readable, _, _ = select.select([stdin], [], [], 0.0)
        if not readable:
            break
        chars.append(stdin.read(1))

    if not chars:
        return None
    seq = "".join(chars)
    if "\x1b[A" in seq or "[A" in seq:
        return "UP"
    if "\x1b[B" in seq or "[B" in seq:
        return "DOWN"
    if "s" in seq or "S" in seq:
        return "S"
    return seq[-1]


def run_mujoco(policy, cfg):
    if _MUJOCO_IMPORT_ERROR is not None:
        raise ImportError(
            "MuJoCo sim2sim dependencies are required to run sim2sim_mimic.py. "
            "Install mujoco, mujoco_viewer and glfw, then rerun."
        ) from _MUJOCO_IMPORT_ERROR
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    # 质心偏移：模拟实物质心靠后等，单位 m。X 正=前负=后，Y 正=左负=右，Z 正=上负=下
    com_offset = getattr(cfg.sim_config, 'com_offset', [0., 0., 0.])
    if any(v != 0 for v in com_offset):
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        model.body_ipos[torso_id, 0] += com_offset[0]
        model.body_ipos[torso_id, 1] += com_offset[1]
        model.body_ipos[torso_id, 2] += com_offset[2]
        print(f'[sim2sim] 已应用质心偏移: x={com_offset[0]:.3f} y={com_offset[1]:.3f} z={com_offset[2]:.3f} m')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    foot_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'leg_l6_link'),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'leg_r6_link'),
    ]
    foot_body_ids = [bid for bid in foot_body_ids if bid >= 0]
    waist_body_name = getattr(cfg.asset, "waist_name", "waist_yaw_link")
    waist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, waist_body_name)
    static_com_log_interval = getattr(cfg.sim_config, 'static_com_log_interval', 100)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    mujoco.mj_forward(model, data)
    # 直接在终端读取单键输入，避免额外图形窗口影响 MuJoCo 渲染。
    keyboard_fd = None
    keyboard_old_termios = None
    if sys.stdin.isatty():
        keyboard_fd = sys.stdin.fileno()
        keyboard_old_termios = termios.tcgetattr(keyboard_fd)
        tty.setcbreak(keyboard_fd)
        print(
            f"[sim2sim_mimic] 终端键盘控制已启用：前 {START_MOVING_STEP} 步 BPM=0，"
            "之后 BPM=70；S 切换第一帧保持；键盘方向键 ↑ BPM+5；键盘方向键 ↓ BPM-5"
        )
    else:
        print("[sim2sim_mimic] 当前 stdin 不是终端，仍会尝试读取 MuJoCo Viewer 窗口按键")

    def _restore_keyboard():
        if keyboard_old_termios is not None and keyboard_fd is not None:
            termios.tcsetattr(keyboard_fd, termios.TCSADRAIN, keyboard_old_termios)

    atexit.register(_restore_keyboard)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    raw_action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    delayed_action = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_action = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_q_filter = np.zeros((cfg.env.num_actions), dtype=np.double)
    delayed_target_q_filter = np.zeros((cfg.env.num_actions), dtype=np.double)
    final_target = np.zeros((cfg.env.num_actions), dtype=np.double)

    start_dance = False 
    hold_first_frame = False
    count_lowlevel = 0
    phase_rad = 0.0
    post_start_bpm = float(getattr(cfg.sim_config, "bpm", DEFAULT_BPM))
    bpm_cmd = PRESTART_BPM
    tau_buffer = []
    q_buffer = []
    dq_buffer = []
    dq_for_pd_buffer = []
    q_parallel_buffer = []
    dq_parallel_buffer = []
    obs_euler_buffer = []
    raw_action_buffer = []
    delayed_action_buffer = []
    target_q_filter_buffer = []
    final_target_buffer = []
    ref_dof_buffer = []
    foot_force_buffer = []
    foot_force_source = None

    ankle_dq_filter_indices = np.asarray(
        getattr(cfg.sim_config, "ankle_dq_filter_indices", [4, 5, 10, 11]),
        dtype=np.int64,
    )
    ankle_dq_filter_cutoff_hz = float(getattr(cfg.sim_config, "ankle_dq_filter_cutoff_hz", 10.0))
    if ankle_dq_filter_cutoff_hz > 0.0:
        ankle_dq_filter_alpha = 1.0 - np.exp(-2.0 * np.pi * ankle_dq_filter_cutoff_hz * cfg.sim_config.dt)
    else:
        ankle_dq_filter_alpha = 1.0
    ankle_dq_filtered = np.zeros(len(ankle_dq_filter_indices), dtype=np.float64)
    ankle_dq_filter_initialized = False
    print(
        "[sim2sim_mimic] ankle dq filter: "
        f"indices={ankle_dq_filter_indices.tolist()}, "
        f"cutoff={ankle_dq_filter_cutoff_hz:g}Hz, alpha={ankle_dq_filter_alpha:.6f}"
    )

    left_foot_body_id = foot_body_ids[0] if len(foot_body_ids) > 0 else -1
    right_foot_body_id = foot_body_ids[1] if len(foot_body_ids) > 1 else -1

    # 初始化四连杆参数，后面所有串并联换算都基于这一套几何假设。
    fourbar = init_fourbar_params()
    # default_dof_pos_serial = np.array([-0.185, 0.0, 0.0, 0.36, -0.175, 0.0, \
    #                                    -0.185, 0.0, 0.0, 0.36, -0.175, 0.0, \
    #                                    0.0, \
    #                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    #                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    #                                    0.0, 0.0], dtype=np.float64)
    default_dof_pos_serial = np.array([ -0.457, 0.192, 0.062, 0.874,  -0.370, -0.126, \
                                       -0.457, -0.192, -0.062, 0.874,  -0.370, 0.126, \
                                       0.0, \
                                       0.171, 0.327, 0.442, -1.093, 1.257, -0.058,  -1.514, \
                                       -0.149, -0.755, 0.306, -1.865, 0.258, 0.155, 1.073, \
                                       0.0, 0.0], dtype=np.float64)
    default_dof_pos_parallel = serial_to_parallel_pos_np(default_dof_pos_serial, fourbar)
    ref_dof_val = default_dof_pos_serial.copy()
    set_initial_joint_state(model, data, default_dof_pos_serial, default_dof_pos_parallel)
    mujoco.mj_forward(model, data)
    print("[sim2sim_mimic] 已按 joint 名设置 MuJoCo 初始关节角，避免 qpos 顺序和 actuator 顺序错位")

    render_index = 0 
    slow_down_factor = 1.0
    reference_model_path = getattr(
        cfg.sim_config,
        "reference_model_path",
        f"{LEGGED_GYM_ROOT_DIR}/BPM_dance/reference_state_keypoint_model.pt",
    )
    reference_generator = ReferenceMotionGenerator(reference_model_path)
    print(f"[sim2sim_mimic] 已加载参考动作生成网络: {reference_model_path}")
    print(f"[sim2sim_mimic] 前 {START_MOVING_STEP} 步 BPM=0，启动后 BPM={post_start_bpm:g}")
    print(f"[sim2sim_mimic] 当前 BPM: {bpm_cmd:g}")
    initial_base_yaw = None

    blend_duration_steps = max(1, int(round(0.8 / cfg.sim_config.dt)))
    blend_to_first_frame = False
    blend_request = False
    blend_start_step = 0
    blend_from_goal_buf = None
    blend_from_ref_dof_val = None
    hold_ref_dof_val, hold_goal_buf = reference_generator.build_reference(
        bpm_cmd, 0.0, default_dof_pos_serial, cfg.env.num_control, zero_ref_motion=True
    )

    def _set_bpm(new_bpm):
        nonlocal bpm_cmd, hold_ref_dof_val, hold_goal_buf
        old_bpm = bpm_cmd
        bpm_cmd = float(np.clip(new_bpm, BPM_MIN, BPM_MAX))
        if abs(bpm_cmd - old_bpm) < 1e-6:
            print(f"[sim2sim_mimic] BPM 已在边界: {bpm_cmd:g}", flush=True)
            return
        hold_ref_dof_val, hold_goal_buf = reference_generator.build_reference(
            bpm_cmd, 0.0, default_dof_pos_serial, cfg.env.num_control, zero_ref_motion=True
        )
        print(f"[sim2sim_mimic] BPM: {old_bpm:g} -> {bpm_cmd:g}", flush=True)

    delay_steps = cfg.sim_config.action_delay
    action_delay_buffer = np.zeros((cfg.env.num_actions, delay_steps + 1), dtype=np.double)
    last_glfw_up_pressed = False
    last_glfw_down_pressed = False
    last_glfw_s_pressed = False
    bpm_log_interval_steps = max(1, int(round(BPM_LOG_INTERVAL_SEC / cfg.sim_config.dt)))

    def _toggle_first_frame_hold():
        nonlocal hold_first_frame, blend_to_first_frame, blend_request
        if hold_first_frame or blend_to_first_frame or blend_request:
            hold_first_frame = False
            blend_to_first_frame = False
            blend_request = False
            print("[sim2sim_mimic] 已恢复参考轨迹正常推进")
        else:
            blend_request = True
            print("[sim2sim_mimic] 已触发平滑切换：0.8 秒内过渡到轨迹第一帧并保持不动")

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        step_start = time.time()
        if count_lowlevel % bpm_log_interval_steps == 0:
            print(f"[sim2sim_mimic] 当前 BPM: {bpm_cmd:g}", flush=True)

        if keyboard_fd is not None:
            key = _read_terminal_key(sys.stdin)
            if key == "S":
                _toggle_first_frame_hold()
            elif key == "UP":
                if start_dance:
                    _set_bpm(bpm_cmd + BPM_STEP)
            elif key == "DOWN":
                if start_dance:
                    _set_bpm(bpm_cmd - BPM_STEP)

        viewer_window = getattr(viewer, "window", None)
        if viewer_window is not None:
            glfw_up_pressed = glfw.get_key(viewer_window, glfw.KEY_UP) == glfw.PRESS
            glfw_down_pressed = glfw.get_key(viewer_window, glfw.KEY_DOWN) == glfw.PRESS
            glfw_s_pressed = glfw.get_key(viewer_window, glfw.KEY_S) == glfw.PRESS
            if glfw_up_pressed and not last_glfw_up_pressed:
                if start_dance:
                    _set_bpm(bpm_cmd + BPM_STEP)
            if glfw_down_pressed and not last_glfw_down_pressed:
                if start_dance:
                    _set_bpm(bpm_cmd - BPM_STEP)
            if glfw_s_pressed and not last_glfw_s_pressed:
                _toggle_first_frame_hold()
            last_glfw_up_pressed = glfw_up_pressed
            last_glfw_down_pressed = glfw_down_pressed
            last_glfw_s_pressed = glfw_s_pressed
        
        # 1. 获取仿真器状态
        _, _, quat, _, omega, gvec = get_obs(data) 
        # 从并联 XML 读回电机状态后，先转成策略使用的并联语义。
        q_xml = np.array(data.actuator_length, dtype=np.float64)
        dq_xml = np.array(data.actuator_velocity, dtype=np.float64)
        q_parallel = parallel_xml_to_policy_pos_np(q_xml)
        dq_parallel = parallel_xml_to_policy_vel_np(dq_xml)
        # 再把并联踝状态还原成串联踝状态，保证旧模型看到的是训练时同一套语义。
        q = parallel_to_serial_pos_np(q_parallel, fourbar)
        dq = parallel_to_serial_vel_np(q, q_parallel, dq_parallel, fourbar)
        if not ankle_dq_filter_initialized:
            ankle_dq_filtered[:] = dq[ankle_dq_filter_indices]
            ankle_dq_filter_initialized = True
        else:
            ankle_dq_filtered[:] = (
                (1.0 - ankle_dq_filter_alpha) * ankle_dq_filtered
                + ankle_dq_filter_alpha * dq[ankle_dq_filter_indices]
            )
        dq_for_pd = dq.copy()
        dq_for_pd[ankle_dq_filter_indices] = ankle_dq_filtered
        base_euler_xyz = R.from_quat(quat).as_euler("xyz")
        obs_euler_xyz = base_euler_xyz.copy()
        if start_dance and initial_base_yaw is not None:
            obs_euler_xyz[2] = wrap_to_pi_np(obs_euler_xyz[2] - initial_base_yaw)
        else:
            obs_euler_xyz[2] = 0.0

        if (not start_dance) and static_com_log_interval > 0 and (count_lowlevel % static_com_log_interval == 0):
            com_world = compute_world_com(model, data)
            feet_mid = compute_body_midpoint(data, foot_body_ids)
            waist_world = get_body_world_pos(data, waist_body_id)
            if feet_mid is not None:
                com_rel_feet = com_world - feet_mid
                msg = (
                    f"[sim2sim][static] step={count_lowlevel} "
                    f"COM_world=({com_world[0]:.4f}, {com_world[1]:.4f}, {com_world[2]:.4f}) "
                    f"feet_mid=({feet_mid[0]:.4f}, {feet_mid[1]:.4f}, {feet_mid[2]:.4f}) "
                    f"COM_rel_feet=({com_rel_feet[0]:.4f}, {com_rel_feet[1]:.4f}, {com_rel_feet[2]:.4f})"
                )
                if waist_world is not None:
                    waist_rel_feet = waist_world - feet_mid
                    msg += (
                        f" waist_world=({waist_world[0]:.4f}, {waist_world[1]:.4f}, {waist_world[2]:.4f})"
                        f" waist_rel_feet=({waist_rel_feet[0]:.4f}, {waist_rel_feet[1]:.4f}, {waist_rel_feet[2]:.4f})"
                    )
                print(msg)
            else:
                msg = (
                    f"[sim2sim][static] step={count_lowlevel} "
                    f"COM_world=({com_world[0]:.4f}, {com_world[1]:.4f}, {com_world[2]:.4f})"
                )
                if waist_world is not None:
                    msg += f" waist_world=({waist_world[0]:.4f}, {waist_world[1]:.4f}, {waist_world[2]:.4f})"
                print(msg)
        
        # 100Hz 策略推理频率
        if count_lowlevel % cfg.sim_config.decimation == 0:
            
            last_action[:] = action[:]

            force_static_goal = (not start_dance) or hold_first_frame or blend_to_first_frame
            if start_dance and not force_static_goal:
                phase_rad = (phase_rad + 2.0 * np.pi * bpm_cmd / 60.0 * cfg.sim_config.dt * cfg.sim_config.decimation) % (2.0 * np.pi)
            num_ctrl = cfg.env.num_control  # 双腿 12 个受控关节
            ref_dof_val, goal_buf = reference_generator.build_reference(
                bpm_cmd,
                phase_rad,
                default_dof_pos_serial,
                cfg.env.num_control,
                zero_ref_motion=force_static_goal,
            )

            if blend_request and (not hold_first_frame) and (not blend_to_first_frame):
                blend_request = False
                blend_to_first_frame = True
                blend_start_step = count_lowlevel
                blend_from_goal_buf = goal_buf.copy()
                blend_from_ref_dof_val = ref_dof_val.copy()

            if blend_to_first_frame:
                alpha = np.clip((count_lowlevel - blend_start_step) / blend_duration_steps, 0.0, 1.0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                goal_buf = ((1.0 - alpha) * blend_from_goal_buf + alpha * hold_goal_buf).astype(np.float32)
                ref_dof_val = ((1.0 - alpha) * blend_from_ref_dof_val + alpha * hold_ref_dof_val).astype(np.float64)
                if alpha >= 1.0 - 1e-6:
                    blend_to_first_frame = False
                    hold_first_frame = True
                    print("[sim2sim_mimic] 已完成平滑切换，当前保持轨迹第一帧不动")
            elif hold_first_frame:
                goal_buf = hold_goal_buf.copy()
                ref_dof_val = hold_ref_dof_val.copy()

            # --- 构建 Proprioception：关节位置观测使用 q - 当前参考关节角 ---
            control_idx = np.asarray(cfg.env.num_control, dtype=np.int64)
            n_ctrl = len(control_idx)
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            obs[0, 0:n_ctrl] = (q[control_idx] - ref_dof_val[control_idx]) * 1.0
            obs[0, n_ctrl:2 * n_ctrl] = dq[control_idx] * 0.05
            obs[0, 2 * n_ctrl:3 * n_ctrl] = action[control_idx]
            offset = 3 * n_ctrl
            obs[0, offset:offset + 3] = omega * 1.0
            obs[0, offset + 3:offset + 6] = obs_euler_xyz * 1.0
            obs[0, offset + 6:offset + 7] = math.sin(phase_rad)
            obs[0, offset + 7:offset + 8] = math.cos(phase_rad)
            obs[0, offset + 8:offset + 9] = reference_generator.normalized_bpm(bpm_cmd)

            # --- 拼接 Policy Input: 当前 obs(45) + 当前 goal(19) = 64 ---
            policy_input = np.concatenate([obs.reshape(-1), goal_buf]).reshape(1, -1).astype(np.float32)
            if goal_buf.shape[0] != cfg.env.num_goal_obs:
                raise RuntimeError(f"goal 维度错误: got {goal_buf.shape[0]}, expected {cfg.env.num_goal_obs}")
            if policy_input.shape[1] != cfg.env.num_observations:
                raise RuntimeError(
                    f"policy 输入维度错误: got {policy_input.shape[1]}, "
                    f"expected {cfg.env.num_observations}"
                )
            
            # --- Inference ---
            raw_action = np.zeros((cfg.env.num_actions), dtype=np.double)
            
            # 1. RL 输出 (受控关节)
            with torch.no_grad():
                rl_out = policy(torch.tensor(policy_input))[0].numpy()
            if len(rl_out) != n_ctrl:
                raise RuntimeError(f"策略输出维度错误: got {len(rl_out)}, expected {n_ctrl}")
            raw_action[cfg.env.num_control] = rl_out
            
            # 2. 参考轨迹 (非受控关节)
            # 这里对应非受控关节，直接跟随 ref
            raw_action[cfg.env.num_notcontrol] = ref_dof_val[cfg.env.ref_num_notcontrol] / cfg.control.action_scale
            
            # 3. Clip，并保存当前策略命令（训练环境中的 self.actions）
            raw_action = np.clip(raw_action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            action[:] = raw_action

        # 滤波器计算 target_q_filter（当前未延迟的目标增量）
        if cfg.normalization.actions_filter:
            rate_ = (count_lowlevel % cfg.sim_config.decimation + 1.) / cfg.sim_config.decimation
            action_filter = (1. - rate_) * last_action + rate_ * action
            target_q_filter = action_filter * cfg.control.action_scale
        else:
            target_q_filter = action * cfg.control.action_scale

        # Action Delay：按低层 step 推进，和训练环境语义一致
        if delay_steps > 0:
            action_delay_buffer[:, 1:] = action_delay_buffer[:, :-1]
            action_delay_buffer[:, 0] = target_q_filter.copy()
            delayed_target_q_filter = action_delay_buffer[:, delay_steps]
        else:
            delayed_target_q_filter = target_q_filter.copy()

        if cfg.control.action_scale != 0:
            delayed_action = delayed_target_q_filter / cfg.control.action_scale
        else:
            delayed_action[:] = delayed_target_q_filter

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # 非受控关节始终显式跟随参考轨迹，避免在 default_dof_pos != 0 时重复叠加默认角。
        final_target = ref_dof_val.copy()
        if getattr(cfg.control, "use_ref_residual_target", False):
            final_target[num_ctrl] = ref_dof_val[num_ctrl] + delayed_target_q_filter[num_ctrl]
        else:
            final_target[num_ctrl] = delayed_target_q_filter[num_ctrl] + default_dof_pos_serial[num_ctrl]

        # 先在串联关节空间做 PD，再把踝关节力矩映射成并联电机力矩下发给 XML。
        tau_serial = pd_control(final_target, q, cfg.robot_config.kps,
                                target_dq, dq_for_pd, cfg.robot_config.kds)
        tau_parallel = serial_tau_to_parallel_policy_tau_np(tau_serial, q, q_parallel, fourbar)
        tau = policy_parallel_to_xml_tau_np(tau_parallel)

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        
        # 记录数据用于绘图（全程记录）
        tau_buffer.append(tau_parallel.copy())
        q_buffer.append(q.copy())
        dq_buffer.append(dq.copy())
        dq_for_pd_buffer.append(dq_for_pd.copy())
        q_parallel_buffer.append(q_parallel.copy())
        dq_parallel_buffer.append(dq_parallel.copy())
        obs_euler_buffer.append(obs_euler_xyz.copy())
        raw_action_buffer.append(raw_action.copy())
        delayed_action_buffer.append(delayed_action.copy())
        target_q_filter_buffer.append(target_q_filter.copy())
        final_target_buffer.append(final_target.copy())
        ref_dof_buffer.append(ref_dof_val.copy())
        
        if(render_index % 2 == 0):
            viewer.render()

        mujoco.mj_step(model, data)

        lf, rf, fsrc = get_foot_forces(model, data, left_foot_body_id, right_foot_body_id)
        if foot_force_source is None:
            foot_force_source = fsrc
        foot_force_buffer.append(np.concatenate([lf, rf]).astype(np.float64))

        count_lowlevel += 1
        if count_lowlevel == START_MOVING_STEP:
            _set_bpm(post_start_bpm)
            print(f'--- Start Moving Reference (Step {START_MOVING_STEP}, BPM={bpm_cmd:g}) ---')
            # 在真正开始推进参考动作的瞬间锁定一次初始 yaw，
            # 后续观测中的 yaw 都相对这个时刻计算偏差。
            initial_base_yaw = float(base_euler_xyz[2])
            start_dance = True

        render_index += 1
        
        elapsed = time.time() - step_start
        target_duration = cfg.sim_config.dt * slow_down_factor
        # 只有单步耗时超过 100 Hz 预算（0.01s）时才打印，避免正常情况下刷屏。
        if elapsed > 0.01:
            print(f"[sim2sim] Step {count_lowlevel}: Simulated in {elapsed:.3f}s, exceeded 100Hz budget.")
        if target_duration > elapsed:
            time.sleep(target_duration - elapsed)

    _restore_keyboard()
    viewer.close()
    _setup_matplotlib_chinese_font()
    print("Plotting Torques...")
    plot_start_idx = 1000 if len(tau_buffer) > 1000 else 0
    tau_data = np.array(tau_buffer)[plot_start_idx:]
    q_data = np.array(q_buffer)[plot_start_idx:]
    dq_data = np.array(dq_buffer)[plot_start_idx:]
    dq_for_pd_data = np.array(dq_for_pd_buffer)[plot_start_idx:]
    q_parallel_data = np.array(q_parallel_buffer)[plot_start_idx:]
    dq_parallel_data = np.array(dq_parallel_buffer)[plot_start_idx:]
    obs_euler_data = np.array(obs_euler_buffer)[plot_start_idx:]
    raw_action_data = np.array(raw_action_buffer)[plot_start_idx:]
    delayed_action_data = np.array(delayed_action_buffer)[plot_start_idx:]
    target_q_filter_data = np.array(target_q_filter_buffer)[plot_start_idx:]
    final_target_data = np.array(final_target_buffer)[plot_start_idx:]
    ref_dof_data = np.array(ref_dof_buffer)[plot_start_idx:]
    foot_force_data = np.array(foot_force_buffer)[plot_start_idx:]
    time_axis = np.arange(tau_data.shape[0]) * cfg.sim_config.dt


    labels_left = ['L_Hip_Pitch', 'L_Hip_Roll', 'L_Hip_Yaw', 'L_Knee', 'L_Ankle_Pitch', 'L_Ankle_Roll']
    labels_right = ['R_Hip_Pitch', 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Knee', 'R_Ankle_Pitch', 'R_Ankle_Roll']

    def plot_joint_groups(series_data, figure_title, y_label):
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        for i in range(6):
            axs[0].plot(time_axis, series_data[:, i], label=labels_left[i] if i < len(labels_left) else f'Joint {i}')
        axs[0].set_title(f"{figure_title} - Left Leg")
        axs[0].set_ylabel(y_label)
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        for i in range(6):
            idx = i + 6
            axs[1].plot(time_axis, series_data[:, idx], label=labels_right[i] if i < len(labels_right) else f'Joint {idx}')
        axs[1].set_title(f"{figure_title} - Right Leg")
        axs[1].set_ylabel(y_label)
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        axs[2].plot(time_axis, series_data[:, 12], label='Waist', linewidth=2, color='black')
        axs[2].set_title(f"{figure_title} - Waist")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel(y_label)
        axs[2].legend(loc='upper right')
        axs[2].grid(True)
        plt.tight_layout()
        return fig, axs

    def plot_control_pipeline_group(indices, labels, figure_title):
        fig, axs = plt.subplots(len(indices), 1, figsize=(14, 3.2 * len(indices)), sharex=True)
        if len(indices) == 1:
            axs = [axs]
        for ax, idx, label in zip(axs, indices, labels):
            ax.plot(time_axis, raw_action_data[:, idx], label='raw_action', linewidth=1.0)
            ax.plot(time_axis, delayed_action_data[:, idx], label='delayed_action', linewidth=1.0)
            ax.plot(time_axis, target_q_filter_data[:, idx], label='filtered_target_delta', linewidth=1.0)
            ax.plot(time_axis, final_target_data[:, idx], label='final_pd_target', linewidth=1.2)
            ax.plot(time_axis, ref_dof_data[:, idx], label='ref_dof', linewidth=1.2, linestyle='--')
            ax.set_title(label)
            ax.set_ylabel("rad")
            ax.grid(True)
        axs[0].legend(loc='upper right', ncol=5)
        axs[-1].set_xlabel("Time (s)")
        fig.suptitle(figure_title, fontsize=14)
        plt.tight_layout()
        return fig, axs

    def plot_ankle_pitch_diagnostics():
        ankle_indices = [4, 10, 5, 11]
        ankle_labels = ['L_Ankle_Pitch', 'R_Ankle_Pitch', 'L_Ankle_Roll', 'R_Ankle_Roll']
        fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        for ax, idx, label in zip(axs, ankle_indices, ankle_labels):
            ax.plot(time_axis, q_data[:, idx], label='q', linewidth=1.3, color='C0')
            ax.plot(time_axis, ref_dof_data[:, idx], label='ref_dof', linewidth=1.5, linestyle='--')
            ax.plot(time_axis, final_target_data[:, idx], label='final_pd_target', linewidth=1.5, color='C1')
            ax.set_title(f"{label} - q / ref_dof / final_pd_target / tau")
            ax.set_ylabel("rad")
            ax.grid(True)
            ax_right = ax.twinx()
            ax_right.plot(time_axis, tau_data[:, idx], label='tau', linewidth=1.1, color='C3', alpha=0.8)
            ax_right.set_ylabel("tau")
            lines_left, labels_left_ = ax.get_legend_handles_labels()
            lines_right, labels_right_ = ax_right.get_legend_handles_labels()
            ax.legend(lines_left + lines_right, labels_left_ + labels_right_, loc='upper right')
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig, axs

    def plot_ankle_parallel_serial_velocity():
        groups = [
            ("Left Ankle", [4, 5], ["serial_pitch", "serial_roll"], ["parallel_motor_4_2", "parallel_motor_4_1"]),
            ("Right Ankle", [10, 11], ["serial_pitch", "serial_roll"], ["parallel_motor_4_2", "parallel_motor_4_1"]),
        ]
        fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for ax, (title, indices, serial_labels, parallel_labels) in zip(axs, groups):
            for idx, label in zip(indices, serial_labels):
                ax.plot(time_axis, dq_data[:, idx], label=f"{label} dq_serial_raw", linewidth=0.9, alpha=0.55)
                ax.plot(time_axis, dq_for_pd_data[:, idx], label=f"{label} dq_for_pd_filtered", linewidth=1.4)
            for idx, label in zip(indices, parallel_labels):
                ax.plot(time_axis, dq_parallel_data[:, idx], label=f"{label} dq_parallel", linewidth=1.0, linestyle="--")
            ax.set_title(f"{title} Velocity: Serial Equivalent vs Parallel Motors")
            ax.set_ylabel("rad/s")
            ax.grid(True)
            ax.legend(loc='upper right', ncol=2)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig, axs

    def print_ankle_pitch_error_summary():
        ankle_pitch_indices = [4, 10]
        ankle_pitch_labels = ['L_Ankle_Pitch', 'R_Ankle_Pitch']
        window_steps = min(len(q_data), int(5.0 / cfg.sim_config.dt))
        print("Ankle pitch error summary (actual - reference):")
        for idx, label in zip(ankle_pitch_indices, ankle_pitch_labels):
            err_rad = q_data[:, idx] - ref_dof_data[:, idx]
            err_deg = np.rad2deg(err_rad)
            mean_err = np.mean(err_deg)
            mae = np.mean(np.abs(err_deg))
            max_abs = np.max(np.abs(err_deg))
            final_mean = np.mean(err_deg[-window_steps:]) if window_steps > 0 else err_deg[-1]
            final_mae = np.mean(np.abs(err_deg[-window_steps:])) if window_steps > 0 else abs(err_deg[-1])
            print(
                f"  {label}: "
                f"mean={mean_err:+.2f} deg, "
                f"MAE={mae:.2f} deg, "
                f"max_abs={max_abs:.2f} deg, "
                f"last_{window_steps}steps_mean={final_mean:+.2f} deg, "
                f"last_{window_steps}steps_MAE={final_mae:.2f} deg"
            )

    def plot_leg_joint_angle_comparison():
        leg_indices = list(range(12))
        leg_labels = labels_left + labels_right
        fig, axs = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
        axs = axs.flatten()

        for ax, idx, label in zip(axs, leg_indices, leg_labels):
            actual = np.rad2deg(q_data[:, idx])
            reference = np.rad2deg(ref_dof_data[:, idx])
            err = actual - reference
            mae = np.mean(np.abs(err))
            max_err = np.max(np.abs(err))

            ax.plot(time_axis, actual, label='actual_q', linewidth=1.3, color='C0')
            ax.plot(time_axis, reference, label='ref_dof', linewidth=1.3, linestyle='--', color='C1')
            ax.set_title(label)
            ax.set_ylabel("deg")
            ax.grid(True)
            ax.text(
                0.02,
                0.98,
                f"MAE={mae:.2f} deg\nMax={max_err:.2f} deg",
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
            )

        axs[0].legend(loc='upper right')
        axs[-2].set_xlabel("Time (s)")
        axs[-1].set_xlabel("Time (s)")
        fig.suptitle("Leg Joint Angle Comparison: Actual vs Reference", fontsize=16)
        plt.tight_layout()
        return fig, axs

    def plot_obs_euler():
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        labels = ["roll", "pitch", "yaw_rel"]
        for i, label in enumerate(labels):
            axs[i].plot(time_axis, obs_euler_data[:, i], linewidth=1.4)
            axs[i].set_title(f"Observation Euler - {label}")
            axs[i].set_ylabel("rad")
            axs[i].grid(True)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig, axs

    plot_joint_groups(tau_data, "Torques", "Torque")

    print("Plotting Velocities...")
    plot_joint_groups(dq_data, "Velocities", "Velocity")
    plot_ankle_parallel_serial_velocity()

    print("Plotting Observation Euler...")
    plot_obs_euler()

    print("Plotting Actions and Targets...")
    plot_control_pipeline_group(list(range(6)), labels_left, "Control Pipeline - Left Leg")
    plot_control_pipeline_group(list(range(6, 12)), labels_right, "Control Pipeline - Right Leg")
    plot_control_pipeline_group([12], ['Waist'], "Control Pipeline - Waist")
    plot_ankle_pitch_diagnostics()
    print_ankle_pitch_error_summary()

    print("Plotting leg joint angle comparison...")
    plot_leg_joint_angle_comparison()

    plt.show()

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, help='Path to .pt (JIT) or .onnx policy')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--bpm', type=float, default=DEFAULT_BPM, help='BPM used after the first 1000 low-level steps')
    parser.add_argument(
        '--reference_model',
        type=str,
        default=f'{LEGGED_GYM_ROOT_DIR}/BPM_dance/reference_state_keypoint_model.pt',
        help='Path to reference motion generator checkpoint',
    )
    args = parser.parse_args()

    class Sim2simCfg(MrobotMimicCfg):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/Mrobot/mjcf/mjmodel_terrain.xml'
            else:
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/xml/CASBOT_02_shell_ENCOS_7dof_par.xml'
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/xml/CASBOT_02_shell_ENCOS_7dof_par_bass.xml'  # bass
            sim_duration = 95.0
            dt = 0.001
            decimation = 10
            action_delay = 0
            bpm = args.bpm
            reference_model_path = args.reference_model
            static_com_log_interval = 1000  # 静态站立阶段每多少个低层 step 打印一次 COM
            # 质心偏移 [x,y,z] 单位 m，用于模拟实物质心靠后。x 负=后移，如 [-0.03, 0, 0] 表示后移 3cm
            com_offset = [-0.0, 0.0, 0.0]

        class robot_config:
            # kp/kd
            
            # PD 9
            kps = np.array([276.348923229 / 2, 276.348923229 / 2, 256.6097056 / 2, 276.348923229 / 2, 153.965828656 / 2, 153.965828656 / 2, \
                            276.348923229 / 2, 276.348923229 / 2, 256.6097056 / 2, 276.348923229 / 2, 153.965828656 / 2, 153.965828656 / 2, \
                            153.965828656 / 2, \
                            200, 200, 200, 200, 200, 200, 200,\
                            200, 200, 200, 200, 200, 200, 200,\
                            200, 200], dtype=np.double)
            kds = np.array([17.5929188596 / 2, 17.5929188596 / 2, 16.33628152 / 2, 17.5929188596 / 2, 9.80176907892 / 2, 9.80176907892 / 2,  \
                            17.5929188596 / 2, 17.5929188596 / 2, 16.33628152 / 2, 17.5929188596 / 2, 9.80176907892 / 2, 9.80176907892 / 2,  \
                            9.80176907892 / 2, \
                            5, 5, 5, 5, 5, 5, 5, \
                            5, 5, 5, 5, 5, 5, 5, \
                            5, 5], dtype=np.double)
            
            
            tau_limit = np.array([66.7, 86.7, 60.1, 86.7,  31.5,  31.5, \
                                  66.7, 86.7, 60.1, 86.7,  31.5,  31.5, \
                                   35.2, \
                                  35.2,  35.2,  35.2,  35.2,  35.2,  35.2,  35.2, \
                                   35.2,  35.2,  35.2,  35.2,  35.2,  35.2,  35.2, \
                                   35.2,  35.2], dtype=np.double)

            # tau_limit = np.array([77.2, 77.2, 35.2, 77.2,  35.2,  35.2, \
            #                       77.2, 77.2, 35.2, 77.2,  35.2,  35.2, \
            #                       60, 60, 60, 60, 60, 60, 60, \
            #                       60, 60, 60, 60, 60, 60, 60, \
            #                       60, 60, 60], dtype=np.double)
    
    policy = load_policy(args.load_model)
    run_mujoco(policy, Sim2simCfg())
