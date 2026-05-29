#!/usr/bin/env python3
"""Train a reference-state network from BPM/phase keypoint trajectory CSVs.

功能：
    使用经过 Isaac Gym 仿真导出的 keypoint CSV 训练参考量网络。
    网络输入为 BPM + 当前 beat phase，输出当前相位的完整参考量：
        关节角度 *_pos / 关节角速度 *_vel / 身体关键点信息。

输入输出：
    输入：BPM, sin(phase), cos(phase)
    输出：CSV 中除 timestamp / beat_phase_rad / bpm 以外的全部参考列
    默认数据：BPM_dance/bpm_phase_state_dataset_keypoint
    默认模型：BPM_dance/reference_state_keypoint_model.pt
    默认 ONNX：BPM_dance/reference_state_keypoint_model.onnx

运行指令：
    推荐使用项目 conda 环境：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/train_reference_state_network.py --epochs 1500 --device cuda

    CPU 快速测试：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/train_reference_state_network.py --epochs 1 --device cpu

    自定义保存路径：
        /home/weil/anaconda3/envs/hl_rl/bin/python BPM_dance/train_reference_state_network.py \
            --epochs 1500 --device cuda \
            --output BPM_dance/reference_state_keypoint_model.pt

备注：
    phase 会在网络内部编码为 sin(phase), cos(phase)，避免 0 和 2*pi 之间的不连续。
    导出的 ONNX 输入为 raw_input，shape=(N, 2)，列顺序是 [bpm, phase_rad]。
    导出的 ONNX 输出为 reference_state，shape=(N, 输出维度)，已经是反归一化后的真实参考量。
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "BPM_dance" / "bpm_phase_state_dataset_keypoint"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "BPM_dance" / "reference_state_keypoint_model.pt"
NON_TARGET_COLUMNS = {"timestamp", "beat_phase_rad", "bpm"}
STATIC_BPM_FILE_NAME = "bpm_000_keypoint.csv"
DEFAULT_STATIC_REPEAT = 20
DEFAULT_STATIC_LOSS_WEIGHT = 10.0
DEFAULT_TRAIN_BPMS = "0,60:170"


def _parse_hidden(value: str) -> List[int]:
    try:
        hidden = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden sizes must be comma-separated integers") from exc
    if not hidden or any(v <= 0 for v in hidden):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return hidden


def _parse_bpm_values(value: str) -> List[int]:
    """Parse comma-separated BPM values and inclusive ranges, e.g. 0,60:170."""
    bpms = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                parts = [int(v.strip()) for v in item.split(":")]
                if len(parts) not in (2, 3):
                    raise ValueError
                start, stop = parts[:2]
                step = parts[2] if len(parts) == 3 else 1
                if step <= 0:
                    raise ValueError
                bpms.extend(range(start, stop + 1, step))
            else:
                bpms.append(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "BPM values must look like '0,60:170' or '0,60:170:5'"
        ) from exc
    if not bpms:
        raise argparse.ArgumentTypeError("At least one BPM value is required")
    return sorted(set(bpms))


def _bpm_from_filename(path: Path) -> float:
    match = re.search(r"bpm_(\d+(?:\.\d+)?)", path.stem)
    if not match:
        raise ValueError(f"Could not parse BPM from filename: {path.name}")
    return float(match.group(1))


def _read_header(path: Path) -> List[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _select_target_columns(header: Sequence[str]) -> List[str]:
    return [col for col in header if col not in NON_TARGET_COLUMNS]


def _path_for_bpm(data_dir: Path, bpm: int) -> Path:
    return data_dir / f"bpm_{bpm:03d}_keypoint.csv"


def _load_dataset(
    data_dir: Path,
    bpm_values: Sequence[int],
    static_repeat: int = DEFAULT_STATIC_REPEAT,
    static_loss_weight: float = DEFAULT_STATIC_LOSS_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[float]]:
    csv_paths = [_path_for_bpm(data_dir, bpm) for bpm in bpm_values]
    missing_paths = [path for path in csv_paths if not path.exists()]
    csv_paths = [path for path in csv_paths if path.exists()]
    if not csv_paths:
        raise FileNotFoundError(f"No requested BPM csv files found in {data_dir}")
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths[:8])
        if len(missing_paths) > 8:
            missing_names += f", ... ({len(missing_paths)} missing total)"
        print(f"Warning: missing requested BPM files: {missing_names}")

    all_inputs = []
    all_targets = []
    all_weights = []
    loaded_bpms = []
    output_columns = None

    for path in csv_paths:
        bpm = _bpm_from_filename(path)
        loaded_bpms.append(bpm)
        is_static_file = path.name == STATIC_BPM_FILE_NAME
        header = _read_header(path)
        if "beat_phase_rad" not in header:
            raise ValueError(f"{path} has no beat_phase_rad column")

        phase_idx = header.index("beat_phase_rad")
        target_cols = _select_target_columns(header)
        if not target_cols:
            raise ValueError(f"{path} has no reference target columns")
        if output_columns is None:
            output_columns = target_cols
        elif output_columns != target_cols:
            raise ValueError(f"{path} output columns do not match previous files")

        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]

        phases = data[:, phase_idx]
        target_indices = [header.index(col) for col in target_cols]
        targets = data[:, target_indices]

        inputs = np.column_stack([
            np.full(len(phases), bpm, dtype=np.float32),
            phases.astype(np.float32),
        ])
        repeat = static_repeat if is_static_file else 1
        if repeat <= 0:
            continue
        if repeat > 1:
            print(f"Repeating static BPM file {path.name}: {repeat}x")
            inputs = np.tile(inputs, (repeat, 1))
            targets = np.tile(targets, (repeat, 1))
        weight = float(static_loss_weight) if is_static_file else 1.0
        if is_static_file:
            print(f"Static BPM loss weight for {path.name}: {weight:g}")
        all_inputs.append(inputs)
        all_targets.append(targets.astype(np.float32))
        all_weights.append(np.full((len(inputs),), weight, dtype=np.float32))

    assert output_columns is not None
    return (
        np.vstack(all_inputs).astype(np.float32),
        np.vstack(all_targets).astype(np.float32),
        np.concatenate(all_weights).astype(np.float32),
        output_columns,
        loaded_bpms,
    )


def _encode_inputs(raw_inputs: np.ndarray, bpm_mean: float, bpm_std: float) -> np.ndarray:
    bpm = (raw_inputs[:, 0:1] - bpm_mean) / bpm_std
    phase = raw_inputs[:, 1:2]
    return np.concatenate([bpm, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)


def predict_reference_state(
    model: nn.Module,
    bpm: float,
    phase_rad: np.ndarray,
    checkpoint: Dict,
    device: torch.device,
) -> np.ndarray:
    """Predict denormalized reference columns for BPM + phase(rad)."""
    phase = np.asarray(phase_rad, dtype=np.float32).reshape(-1)
    raw_inputs = np.column_stack([
        np.full(len(phase), bpm, dtype=np.float32),
        phase,
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


class ReferenceStateNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for width in hidden:
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.SiLU())
            last_dim = width
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReferenceStateOnnxWrapper(nn.Module):
    """ONNX wrapper with raw [bpm, phase_rad] input and denormalized output."""

    def __init__(
        self,
        model: ReferenceStateNet,
        bpm_mean: float,
        bpm_std: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("bpm_mean", torch.tensor(float(bpm_mean), dtype=torch.float32))
        self.register_buffer("bpm_std", torch.tensor(float(bpm_std), dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        bpm = raw_input[:, 0:1]
        phase = raw_input[:, 1:2]
        encoded = torch.cat([
            (bpm - self.bpm_mean) / self.bpm_std,
            torch.sin(phase),
            torch.cos(phase),
        ], dim=1)
        pred_norm = self.model(encoded)
        return pred_norm * self.target_std + self.target_mean


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _plot_loss_curve(history: Sequence[Tuple[int, float, float]], output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hist = np.asarray(history, dtype=np.float64)
    epochs = hist[:, 0]
    train_loss = hist[:, 1]
    val_loss = hist[:, 2]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="train MSE", linewidth=1.8)
    ax.plot(epochs, val_loss, label="val MSE", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("Reference State Network Training Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _export_onnx(
    model: ReferenceStateNet,
    output_path: Path,
    bpm_mean: float,
    bpm_std: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ReferenceStateOnnxWrapper(
        model,
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        target_mean=target_mean,
        target_std=target_std,
    ).cpu().eval()
    dummy_input = torch.tensor([[70.0, 0.0], [120.0, 3.1415926]], dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["raw_input"],
        output_names=["reference_state"],
        dynamic_axes={
            "raw_input": {0: "batch"},
            "reference_state": {0: "batch"},
        },
    )


def _write_onnx_metadata(
    output_path: Path,
    output_columns: Sequence[str],
    loaded_bpms: Sequence[float],
) -> Path:
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = {
        "onnx_file": output_path.name,
        "input_name": "raw_input",
        "input_columns": ["bpm", "phase_rad"],
        "input_shape": ["batch", 2],
        "output_name": "reference_state",
        "output_columns": list(output_columns),
        "output_shape": ["batch", len(output_columns)],
        "loaded_bpms": list(loaded_bpms),
        "note": "ONNX includes BPM normalization, sin/cos phase encoding, and output denormalization.",
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BPM/phase -> full reference state/keypoint network."
    )
    parser.add_argument("--data-dir", type=Path,
                        default=DEFAULT_DATA_DIR,
                        help="Directory containing bpm_*_keypoint.csv files")
    parser.add_argument("-o", "--output", type=Path,
                        default=DEFAULT_OUTPUT_PATH,
                        help="Path to save trained model checkpoint")
    parser.add_argument("--plot-output", type=Path,
                        default=None,
                        help="Path to save train/val loss curve PNG")
    parser.add_argument("--onnx-output", type=Path,
                        default=None,
                        help="Path to save ONNX model (default: output path with .onnx)")
    parser.add_argument("--no-onnx", action="store_true",
                        help="Do not export ONNX after training")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="Training epochs (default: 1500)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (default: 1024)")
    parser.add_argument("--bpm-values", type=_parse_bpm_values,
                        default=_parse_bpm_values(DEFAULT_TRAIN_BPMS),
                        help="BPM files to load, supports inclusive ranges (default: 0,60:170)")
    parser.add_argument("--static-repeat", type=int, default=DEFAULT_STATIC_REPEAT,
                        help="Repeat bpm_000_keypoint.csv this many times during loading (default: 20)")
    parser.add_argument("--static-loss-weight", type=float, default=DEFAULT_STATIC_LOSS_WEIGHT,
                        help="Per-sample loss weight for bpm_000_keypoint.csv after repeat (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--hidden", type=_parse_hidden, default=[256, 256, 256],
                        help="MLP hidden sizes, comma-separated (default: 256,256,256)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Validation fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, or cuda (default: auto)")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print metrics every N epochs (default: 100)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw_inputs, targets, sample_weights, output_columns, loaded_bpms = _load_dataset(
        args.data_dir,
        args.bpm_values,
        args.static_repeat,
        args.static_loss_weight,
    )
    bpm_mean = float(raw_inputs[:, 0].mean())
    bpm_std = float(raw_inputs[:, 0].std())
    if bpm_std < 1e-6:
        bpm_std = 1.0

    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0)
    target_std[target_std < 1e-6] = 1.0

    inputs = _encode_inputs(raw_inputs, bpm_mean, bpm_std)
    norm_targets = ((targets - target_mean) / target_std).astype(np.float32)

    train_idx, val_idx = _split_indices(len(inputs), args.val_fraction, args.seed)
    train_ds = TensorDataset(
        torch.from_numpy(inputs[train_idx]),
        torch.from_numpy(norm_targets[train_idx]),
        torch.from_numpy(sample_weights[train_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_x = torch.from_numpy(inputs[val_idx]).to(device) if len(val_idx) else None
    val_y = torch.from_numpy(norm_targets[val_idx]).to(device) if len(val_idx) else None

    model = ReferenceStateNet(
        input_dim=inputs.shape[1],
        output_dim=targets.shape[1],
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print(f"Loaded {len(inputs)} samples from {args.data_dir}")
    print(f"Loaded BPM values: {', '.join(f'{bpm:g}' for bpm in loaded_bpms)}")
    print(f"Input dim={inputs.shape[1]}, output dim={targets.shape[1]}, device={device}")
    print(f"Output columns: {len(output_columns)}")
    print(f"Train samples={len(train_idx)}, val samples={len(val_idx)}")
    print(
        f"Sample weight: min={sample_weights.min():.3g}, max={sample_weights.max():.3g}, "
        f"mean={sample_weights.mean():.3g}"
    )

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            per_sample_loss = torch.mean(torch.square(pred - batch_y), dim=1)
            loss = torch.sum(per_sample_loss * batch_w) / torch.sum(batch_w).clamp(min=1e-6)
            loss.backward()
            optimizer.step()

            batch_weight = float(torch.sum(batch_w).item())
            train_loss_sum += float(loss.item()) * batch_weight
            train_weight_sum += batch_weight

        train_loss = train_loss_sum / max(train_weight_sum, 1e-6)
        if val_x is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).item())
        else:
            val_loss = train_loss

        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(f"epoch {epoch:5d}  train_mse={train_loss:.6e}  val_mse={val_loss:.6e}")

    if best_state:
        model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "hidden": list(args.hidden),
        "input_dim": int(inputs.shape[1]),
        "output_dim": int(targets.shape[1]),
        "output_columns": output_columns,
        "input_columns": ["bpm_normalized", "sin_phase", "cos_phase"],
        "target_column_mode": "all_columns_except_timestamp_phase_bpm",
        "data_dir": str(args.data_dir),
        "loaded_bpms": loaded_bpms,
        "requested_bpms": list(args.bpm_values),
        "bpm_mean": bpm_mean,
        "bpm_std": bpm_std,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "sample_weight_mode": {
            "static_file": STATIC_BPM_FILE_NAME,
            "static_repeat": int(args.static_repeat),
            "static_loss_weight": float(args.static_loss_weight),
        },
        "train_args": vars(args),
        "history": history,
    }
    torch.save(checkpoint, args.output)
    plot_output = args.plot_output
    if plot_output is None:
        plot_output = args.output.with_name(args.output.stem + "_loss_curve.png")
    _plot_loss_curve(history, plot_output)
    onnx_output = args.onnx_output
    if onnx_output is None:
        onnx_output = args.output.with_suffix(".onnx")
    if not args.no_onnx:
        _export_onnx(model, onnx_output, bpm_mean, bpm_std, target_mean, target_std)
        onnx_metadata_output = _write_onnx_metadata(onnx_output, output_columns, loaded_bpms)
    print(f"Saved model: {args.output}")
    print(f"Saved loss curve: {plot_output}")
    if not args.no_onnx:
        print(f"Saved ONNX: {onnx_output}")
        print(f"Saved ONNX metadata: {onnx_metadata_output}")
    print(f"Best normalized val MSE: {best_val:.6e}")


if __name__ == "__main__":
    main()
