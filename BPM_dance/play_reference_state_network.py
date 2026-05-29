#!/usr/bin/env python3
"""Load, query, and validate a BPM/phase reference-state network.

功能：
    加载训练好的 BPM/phase 参考量网络。
    输入 BPM 和当前 beat phase 后，输出当前相位的完整参考量：
        关节角度 *_pos / 关节角速度 *_vel / 身体关键点信息。
    也可以对指定 BPM 的整条 keypoint CSV 做验证，输出预测曲线、误差指标和拟合图。

默认路径：
    模型：BPM_dance/reference_state_keypoint_model.pt
    验证数据：BPM_dance/bpm_phase_state_dataset_keypoint
    输出目录：reference_state_eval

运行指令：
    查询单个当前相位参考量：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/play_reference_state_network.py --bpm 90 --phase 1.57

    验证某个 BPM 的整条轨迹：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/play_reference_state_network.py --bpm 90

    指定模型路径：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/play_reference_state_network.py \
            --model BPM_dance/reference_state_keypoint_model.pt \
            --bpm 90 --phase 1.57

输出：
    reference_state_eval/bpm_090_predictions.csv
    reference_state_eval/bpm_090_metrics.txt
    reference_state_eval/bpm_090_fit.png
    reference_state_eval/bpm_090_base_fit.png
    reference_state_eval/bpm_090_keypoint_fit.png
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_reference_state_network import ReferenceStateNet


DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "BPM_dance" / "bpm_phase_state_dataset_keypoint"
    if (PROJECT_ROOT / "BPM_dance" / "bpm_phase_state_dataset_keypoint").exists()
    else PROJECT_ROOT / "bpm_phase_state_dataset_keypoint"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "BPM_dance" / "reference_state_keypoint_model.pt"
DEFAULT_STATIC_DT = 0.01


def _phase_to_timestamp(phase: np.ndarray, bpm: float, static_dt: float = DEFAULT_STATIC_DT) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    if abs(float(bpm)) < 1e-8:
        return np.arange(len(phase), dtype=np.float32) * float(static_dt)
    return phase / (2.0 * np.pi) * (60.0 / float(bpm))


def _read_header(path: Path) -> List[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _load_true_csv(path: Path, output_columns: List[str], bpm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = _read_header(path)
    if "beat_phase_rad" not in header:
        raise ValueError(f"{path} must contain beat_phase_rad")
    missing = [col for col in output_columns if col not in header]
    if missing:
        raise ValueError(f"{path} is missing model output columns: {missing[:5]}")

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    phase = data[:, header.index("beat_phase_rad")]
    if "timestamp" in header:
        timestamp = data[:, header.index("timestamp")]
    else:
        timestamp = _phase_to_timestamp(phase, bpm)
    target = data[:, [header.index(col) for col in output_columns]]
    return timestamp, phase, target


def _encode_inputs(raw_inputs: np.ndarray, bpm_mean: float, bpm_std: float) -> np.ndarray:
    bpm = (raw_inputs[:, 0:1] - bpm_mean) / bpm_std
    phase = raw_inputs[:, 1:2]
    return np.concatenate([bpm, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)


def _predict(
    model: ReferenceStateNet,
    bpm: float,
    phase: np.ndarray,
    checkpoint: Dict,
    device: torch.device,
) -> np.ndarray:
    raw_inputs = np.column_stack([
        np.full(len(phase), bpm, dtype=np.float32),
        phase.astype(np.float32),
    ])
    encoded = _encode_inputs(
        raw_inputs,
        float(checkpoint["bpm_mean"]),
        float(checkpoint["bpm_std"]),
    )
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(encoded).to(device)).cpu().numpy()
    return pred_norm * checkpoint["target_std"] + checkpoint["target_mean"]


def _compute_metrics(pred: np.ndarray, target: np.ndarray, output_columns: List[str]) -> List[str]:
    err = pred - target
    pos_idx = [i for i, col in enumerate(output_columns) if col.endswith("_pos")]
    vel_idx = [i for i, col in enumerate(output_columns) if col.endswith("_vel")]
    base_idx = [i for i, col in enumerate(output_columns) if col.startswith("base_")]
    keypoint_idx = [
        i for i, col in enumerate(output_columns)
        if i not in set(pos_idx) and i not in set(vel_idx) and i not in set(base_idx)
    ]

    def block_metrics(name: str, idx: List[int]) -> str:
        if not idx:
            return f"{name}: no columns"
        e = err[:, idx]
        return (
            f"{name}: mse={np.mean(e ** 2):.8e}, "
            f"rmse={np.sqrt(np.mean(e ** 2)):.8e}, "
            f"mae={np.mean(np.abs(e)):.8e}, "
            f"max_abs={np.max(np.abs(e)):.8e}"
        )

    lines = [
        block_metrics("all", list(range(err.shape[1]))),
        block_metrics("base_state", base_idx),
        block_metrics("position_rad", pos_idx),
        block_metrics("velocity_rad_per_s", vel_idx),
        block_metrics("body_keypoints", keypoint_idx),
        "",
        "Worst columns by RMSE:",
    ]

    per_col_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    order = np.argsort(per_col_rmse)[::-1]
    for idx in order[:10]:
        lines.append(f"  {output_columns[idx]}: rmse={per_col_rmse[idx]:.8e}")
    return lines


def _print_single_reference(bpm: float, phase: float, pred: np.ndarray, output_columns: List[str]) -> None:
    print(f"\nReference prediction: BPM={bpm:g}, phase={phase:.6f} rad")
    for col, value in zip(output_columns, pred.reshape(-1)):
        print(f"{col}: {value:.6f}")


def _write_predictions(
    path: Path,
    bpm: float,
    timestamp: np.ndarray,
    phase: np.ndarray,
    pred: np.ndarray,
    output_columns: List[str],
) -> None:
    columns = ["bpm", "timestamp", "beat_phase_rad"] + output_columns
    bpm_col = np.full(len(phase), bpm, dtype=np.float32)
    data = np.column_stack([bpm_col, timestamp, phase, pred])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(columns),
        fmt="%.6f",
        comments="",
    )


def _plot_fit(
    path: Path,
    phase: np.ndarray,
    pred: np.ndarray,
    target: Optional[np.ndarray],
    output_columns: List[str],
    joints: List[str],
    title: str,
) -> None:
    cache_dir = path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(phase)
    phase_plot = phase[order]
    pred_plot = pred[order]
    target_plot = target[order] if target is not None else None

    rows = len(joints)
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(3.2 * rows, 4)), squeeze=False)
    for row, joint in enumerate(joints):
        for col_idx, suffix in enumerate(("_pos", "_vel")):
            name = joint + suffix
            ax = axes[row][col_idx]
            if name not in output_columns:
                ax.set_visible(False)
                continue
            idx = output_columns.index(name)
            ax.plot(phase_plot, pred_plot[:, idx], label="pred", linewidth=1.3)
            if target_plot is not None:
                ax.plot(phase_plot, target_plot[:, idx], label="true", linewidth=1.0, alpha=0.75)
            ax.set_title(name)
            ax.set_xlabel("phase rad")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_base_fit(
    path: Path,
    phase: np.ndarray,
    pred: np.ndarray,
    target: Optional[np.ndarray],
    output_columns: List[str],
    title: str,
) -> bool:
    base_columns = [col for col in output_columns if col.startswith("base_")]
    if not base_columns:
        return False

    cache_dir = path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(phase)
    phase_plot = phase[order]
    pred_plot = pred[order]
    target_plot = target[order] if target is not None else None

    rows = int(np.ceil(len(base_columns) / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(2.4 * rows, 4)), squeeze=False)
    axes_flat = axes.flatten()
    for ax, col in zip(axes_flat, base_columns):
        idx = output_columns.index(col)
        ax.plot(phase_plot, pred_plot[:, idx], label="pred", linewidth=1.3)
        if target_plot is not None:
            ax.plot(phase_plot, target_plot[:, idx], label="true", linewidth=1.0, alpha=0.75)
        ax.set_title(col)
        ax.set_xlabel("phase rad")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(base_columns):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _plot_keypoint_fit(
    path: Path,
    phase: np.ndarray,
    pred: np.ndarray,
    target: Optional[np.ndarray],
    output_columns: List[str],
    title: str,
) -> bool:
    keypoint_columns = [
        col for col in output_columns
        if not col.endswith("_pos") and not col.endswith("_vel") and not col.startswith("base_")
    ]
    if not keypoint_columns:
        return False

    selected = keypoint_columns[: min(12, len(keypoint_columns))]
    cache_dir = path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(phase)
    phase_plot = phase[order]
    pred_plot = pred[order]
    target_plot = target[order] if target is not None else None

    rows = int(np.ceil(len(selected) / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(2.4 * rows, 4)), squeeze=False)
    axes_flat = axes.flatten()
    for ax, col in zip(axes_flat, selected):
        idx = output_columns.index(col)
        ax.plot(phase_plot, pred_plot[:, idx], label="pred", linewidth=1.3)
        if target_plot is not None:
            ax.plot(phase_plot, target_plot[:, idx], label="true", linewidth=1.0, alpha=0.75)
        ax.set_title(col)
        ax.set_xlabel("phase rad")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(selected):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _default_joints(output_columns: List[str]) -> List[str]:
    candidates = [
        "left_leg_knee_pitch",
        "right_leg_knee_pitch",
        "waist_yaw",
        "left_leg_pelvic_pitch",
    ]
    return [joint for joint in candidates if joint + "_pos" in output_columns][:4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or play a trained reference-state network."
    )
    parser.add_argument("--model", type=Path,
                        default=DEFAULT_MODEL_PATH,
                        help="Model checkpoint path")
    parser.add_argument("--data-dir", type=Path,
                        default=DEFAULT_DATA_DIR,
                        help="Dataset directory used for validation CSV lookup")
    parser.add_argument("--bpm", type=float, default=90.0,
                        help="BPM to evaluate or generate (default: 90)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional explicit CSV for validation")
    parser.add_argument("--phase", type=float, default=None,
                        help="If set, query exactly this current beat phase in radians")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=PROJECT_ROOT / "reference_state_eval",
                        help="Directory for plot, metrics, and prediction CSV")
    parser.add_argument("--num-phases", type=int, default=301,
                        help="Number of phase samples when no true CSV is available")
    parser.add_argument("--joints", type=str, default="",
                        help="Comma-separated base joint names to plot")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, or cuda (default: auto)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    output_columns = list(checkpoint["output_columns"])
    model = ReferenceStateNet(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden=checkpoint["hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    csv_path = args.csv
    if csv_path is None and args.phase is None:
        csv_candidates = [
            args.data_dir / f"bpm_{int(round(args.bpm)):03d}_keypoint.csv",
            args.data_dir / f"bpm_{int(round(args.bpm)):03d}.csv",
        ]
        csv_path = next((path for path in csv_candidates if path.exists()), None)

    target = None
    if args.phase is not None:
        phase = np.asarray([args.phase], dtype=np.float32)
        timestamp = _phase_to_timestamp(phase, args.bpm)
        print("Single-phase query mode; validation CSV lookup is skipped.")
    elif csv_path is not None:
        timestamp, phase, target = _load_true_csv(csv_path, output_columns, args.bpm)
        print(f"Loaded validation CSV: {csv_path}")
    else:
        phase = np.linspace(0.0, 2.0 * np.pi, args.num_phases, endpoint=False).astype(np.float32)
        timestamp = _phase_to_timestamp(phase, args.bpm)
        print("No validation CSV found; generating predictions over one beat phase cycle.")

    pred = _predict(model, args.bpm, phase, checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"bpm_{int(round(args.bpm)):03d}"
    pred_path = args.output_dir / f"{tag}_predictions.csv"
    plot_path = args.output_dir / f"{tag}_fit.png"
    base_plot_path = args.output_dir / f"{tag}_base_fit.png"
    keypoint_plot_path = args.output_dir / f"{tag}_keypoint_fit.png"
    metrics_path = args.output_dir / f"{tag}_metrics.txt"

    _write_predictions(pred_path, args.bpm, timestamp, phase, pred, output_columns)
    if args.phase is not None:
        _print_single_reference(args.bpm, args.phase, pred[0], output_columns)

    if target is not None:
        metrics = _compute_metrics(pred, target, output_columns)
    else:
        metrics = ["No ground-truth CSV was available; only predictions were generated."]
    metrics_path.write_text("\n".join(metrics) + "\n")

    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    if not joints:
        joints = _default_joints(output_columns)
    saved_joint_plot = False
    saved_base_plot = False
    saved_keypoint_plot = False
    if len(phase) > 1:
        _plot_fit(
            plot_path,
            phase,
            pred,
            target,
            output_columns,
            joints,
            title=f"Reference state fit, BPM={args.bpm:g}",
        )
        saved_joint_plot = True
        saved_base_plot = _plot_base_fit(
            base_plot_path,
            phase,
            pred,
            target,
            output_columns,
            title=f"Base state fit, BPM={args.bpm:g}",
        )
        saved_keypoint_plot = _plot_keypoint_fit(
            keypoint_plot_path,
            phase,
            pred,
            target,
            output_columns,
            title=f"Keypoint fit, BPM={args.bpm:g}",
        )

    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")
    if saved_joint_plot:
        print(f"Saved plot: {plot_path}")
    if saved_base_plot:
        print(f"Saved base plot: {base_plot_path}")
    if saved_keypoint_plot:
        print(f"Saved keypoint plot: {keypoint_plot_path}")
    if target is not None:
        print("\n".join(metrics[:4]))


if __name__ == "__main__":
    main()
