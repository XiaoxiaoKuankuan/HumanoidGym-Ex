"""Play MRobot mimic policies in IsaacLab/IsaacSim (BPM music or trajectory dance)."""

import argparse
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play MRobot mimic in IsaacLab Direct workflow.")
parser.add_argument(
    "--task",
    type=str,
    default="mrobot_music",
    choices=["mrobot_music", "mrobot_dance"],
    help="mrobot_music: BPM reference network; mrobot_dance: .npz trajectory mimic.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Policy path (checkpoint model_*.pt, TorchScript, or ONNX). "
    "If omitted, use --load_run/--checkpoint or zero actions.",
)
parser.add_argument(
    "--load_run",
    type=str,
    default=None,
    help="Run folder name under logs/<experiment_name>/, or path relative to repo root "
    "(e.g. logs/mrobot_dance_isaaclab/Jun02_11-54-25_).",
)
parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint index; -1 = latest model_*.pt.")
parser.add_argument("--reference_model", type=str, default=None, help="BPM reference network (mrobot_music only).")
parser.add_argument("--motion_files", type=str, default=None, help="Comma-separated .npz files (mrobot_dance only).")
parser.add_argument("--steps", "--step", type=int, default=1200, help="Number of simulation control steps.")
parser.add_argument("--seed", type=int, default=5)
parser.add_argument(
    "--real_time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Throttle playback to the environment control dt. Use --no-real_time for fastest rollout.",
)
parser.add_argument("--keep_open", action="store_true", help="Keep the IsaacSim window open after playback finishes.")
parser.add_argument(
    "--enable_domain_randomization",
    action="store_true",
    help="Keep training domain randomization during play (default: off for deterministic rollout).",
)
parser.add_argument(
    "--deterministic_reset",
    action="store_true",
    default=None,
    help="Disable reset pose/yaw/dof noise. Default: on when domain randomization is disabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_gym_ex.envs.robots.mrobot.isaaclab_env import (  # noqa: E402
    MrobotMimicDanceIsaacLabEnv,
    MrobotMimicDanceIsaacLabEnvCfg,
    MrobotMimicIsaacLabEnv,
    MrobotMimicIsaacLabEnvCfg,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_bpm_config_lab import (  # noqa: E402
    MrobotMimicBPMLabCfg,
    MrobotMimicBPMLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config_lab import (  # noqa: E402
    MrobotMimicDanceLabCfg,
    MrobotMimicDanceLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.xbot.isaaclab_vec_env import IsaacLabRslRlVecEnv  # noqa: E402
from humanoid_gym_ex.algo.ppo.actor_critic import ActorCritic  # noqa: E402


def _activation_from_name(name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if isinstance(name, str):
        return activations.get(name.lower(), nn.ELU)()
    if isinstance(name, nn.Module):
        return name
    return nn.ELU()


def _build_actor_critic_from_cfg(env_cfg, policy_cfg, device):
    activation = _activation_from_name(getattr(policy_cfg, "activation", "elu"))
    actor_critic = ActorCritic(
        env_cfg.env.num_observations,
        env_cfg.env.num_privileged_obs,
        env_cfg.env.num_policy_actions,
        num_aux=0,
        actor_hidden_dims=list(policy_cfg.actor_hidden_dims),
        critic_hidden_dims=list(policy_cfg.critic_hidden_dims),
        init_noise_std=np.asarray(policy_cfg.init_noise_std, dtype=np.float32),
        fixed_std=bool(getattr(policy_cfg, "fixed_std", False)),
        activation=activation,
    ).to(device)
    actor_critic.eval()
    return actor_critic


def resolve_policy_path(policy_arg, load_run, checkpoint, experiment_name):
    if policy_arg:
        return policy_arg
    if not load_run:
        return None

    load_path = Path(load_run)
    if not load_path.is_absolute():
        candidate = Path(LEGGED_GYM_ROOT_DIR) / load_path
        if candidate.exists():
            load_path = candidate
        else:
            load_path = Path(LEGGED_GYM_ROOT_DIR) / "logs" / experiment_name / load_run

    if load_path.is_file():
        return str(load_path)
    if not load_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {load_path}")

    if checkpoint == -1:
        models = sorted(load_path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if not models:
            raise FileNotFoundError(f"No model_*.pt in {load_path}")
        return str(models[-1])
    ckpt = load_path / f"model_{checkpoint}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(ckpt)


class CheckpointPolicy:
    """Inference wrapper for training checkpoints saved by OnPolicyRunner."""

    def __init__(self, checkpoint_path, device, env_cfg, policy_cfg):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"{checkpoint_path} is not an OnPolicyRunner training checkpoint.")
        self.actor_critic = _build_actor_critic_from_cfg(env_cfg, policy_cfg, self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.actor_critic.eval()
        obs_normalizer = checkpoint.get("obs_normalizer")
        if obs_normalizer is not None:
            self.obs_mean = obs_normalizer["_mean"].to(self.device)
            self.obs_std = obs_normalizer["_std"].to(self.device)
        else:
            self.obs_mean = None
            self.obs_std = None

    def __call__(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-2)
        return self.actor_critic.act_inference(obs)


class OnnxPolicy:
    """Minimal ONNXRuntime wrapper for exported actor models."""

    def __init__(self, policy_path, device):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX policy playback requires onnxruntime. "
                "Install it or pass a training checkpoint/TorchScript policy."
            ) from exc
        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = torch.device(device)

    def __call__(self, obs):
        actions = self.session.run(None, {self.input_name: obs.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.as_tensor(actions, dtype=torch.float32, device=self.device)


def load_policy(policy_path, device, env_cfg, policy_cfg):
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file does not exist: {path}")
    if path.suffix.lower() == ".onnx":
        print(f"[play_mrobot_isaaclab] Loading ONNX policy: {path}", flush=True)
        return OnnxPolicy(str(path), device)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        checkpoint = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f"[play_mrobot_isaaclab] Loading training checkpoint policy: {path}", flush=True)
        return CheckpointPolicy(str(path), device, env_cfg, policy_cfg)

    try:
        print(f"[play_mrobot_isaaclab] Loading TorchScript policy: {path}", flush=True)
        policy = torch.jit.load(str(path), map_location=device)
        policy.eval()
        return policy
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load policy '{path}'. Supported formats are OnPolicyRunner checkpoints "
            f"(model_*.pt with model_state_dict), TorchScript .pt, and ONNX .onnx."
        ) from exc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)


def _task_setup(task_name):
    if task_name == "mrobot_dance":
        return (
            MrobotMimicDanceIsaacLabEnvCfg(),
            MrobotMimicDanceIsaacLabEnv,
            MrobotMimicDanceLabCfg,
            MrobotMimicDanceLabCfgPPO,
        )
    return (
        MrobotMimicIsaacLabEnvCfg(),
        MrobotMimicIsaacLabEnv,
        MrobotMimicBPMLabCfg,
        MrobotMimicBPMLabCfgPPO,
    )


def main():
    set_seed(args_cli.seed)
    env_cfg, env_class, cfg_class, train_cfg_class = _task_setup(args_cli.task)
    policy_cfg = train_cfg_class.policy

    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.disable_domain_randomization = not args_cli.enable_domain_randomization
    if args_cli.deterministic_reset is None:
        env_cfg.deterministic_reset = env_cfg.disable_domain_randomization
    else:
        env_cfg.deterministic_reset = args_cli.deterministic_reset
    if env_cfg.disable_domain_randomization:
        print("[play_mrobot_isaaclab] Domain randomization disabled.", flush=True)
    if args_cli.reference_model is not None:
        env_cfg.reference_model_path = args_cli.reference_model
    if args_cli.motion_files is not None:
        env_cfg.motion_files = [item.strip() for item in args_cli.motion_files.split(",") if item.strip()]

    direct_env = env_class(env_cfg)
    vec_env = IsaacLabRslRlVecEnv(direct_env)
    obs, _ = vec_env.reset()

    policy_path = resolve_policy_path(
        args_cli.policy,
        args_cli.load_run,
        args_cli.checkpoint,
        train_cfg_class.runner.experiment_name,
    )
    if policy_path:
        policy = load_policy(policy_path, args_cli.device, cfg_class, policy_cfg)
    else:
        policy = None
        print("[play_mrobot_isaaclab] No --policy/--load_run; running zero actions.", flush=True)

    target_step_time = float(env_cfg.sim.dt * env_cfg.decimation)
    for _ in range(args_cli.steps):
        step_start = time.time()
        if policy is None:
            actions = torch.zeros(vec_env.num_envs, env_cfg.action_space, device=vec_env.device)
        else:
            actions = policy(obs)
        obs, _, _, _, _ = vec_env.step(actions)
        if not getattr(args_cli, "headless", False):
            direct_env.sim.render()
        if args_cli.real_time:
            elapsed = time.time() - step_start
            if elapsed < target_step_time:
                time.sleep(target_step_time - elapsed)
    if args_cli.keep_open and not getattr(args_cli, "headless", False):
        print("[play_mrobot_isaaclab] Playback finished. Keeping window open; close IsaacSim to exit.", flush=True)
        while simulation_app.is_running():
            direct_env.sim.render()
            time.sleep(1.0 / 60.0)
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
