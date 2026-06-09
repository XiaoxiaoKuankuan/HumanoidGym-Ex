"""Train MRobot mimic tasks in IsaacLab/IsaacSim with the local PPO runner."""

import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from isaaclab.app import AppLauncher

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="Train MRobot mimic tasks in IsaacLab Direct workflow.")
parser.add_argument("--task", type=str, default="mrobot_music", choices=["mrobot_music", "mrobot_dance"], help="MRobot task name.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--num_steps_per_env", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--reference_model", type=str, default=None)
parser.add_argument("--motion_files", type=str, default=None, help="Comma-separated .npz files for --task mrobot_dance.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--no_log", action="store_true")
parser.add_argument("--disable_domain_randomization", action="store_true", help="Disable IsaacLab MRobot domain randomization for debugging.")
parser.add_argument("--deterministic_reset", action="store_true", help="Disable reset pose/phase noise except BPM sampling.")
parser.add_argument(
    "--profile_step_timings",
    action="store_true",
    help="Synchronize CUDA and print IsaacLab env.step timing breakdown. Use only for profiling.",
)
parser.add_argument(
    "--profile_step_timing_interval",
    type=int,
    default=200,
    help="Number of env.step calls averaged per timing report when --profile_step_timings is enabled.",
)
parser.add_argument(
    "--profile_step_timing_warmup",
    type=int,
    default=20,
    help="Number of initial env.step calls ignored by the timing profiler.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_gym_ex.algo.ppo.on_policy_runner import OnPolicyRunner  # noqa: E402
from humanoid_gym_ex.envs.robots.mrobot.isaaclab_env import (  # noqa: E402
    MrobotMimicDanceIsaacLabEnv,
    MrobotMimicDanceIsaacLabEnvCfg,
    MrobotMimicBPMIsaacLabEnv,
    MrobotMimicBPMIsaacLabEnvCfg,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config_lab import (  # noqa: E402
    MrobotMimicDanceLabCfg,
    MrobotMimicDanceLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_bpm_config_lab import (  # noqa: E402
    MrobotMimicBPMLabCfg,
    MrobotMimicBPMLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.xbot.isaaclab_vec_env import IsaacLabRslRlVecEnv  # noqa: E402


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, list):
            result[key] = [class_to_dict(item) for item in value]
        else:
            result[key] = class_to_dict(value)
    return result


def save_training_config(log_dir, train_cfg):
    save_config = getattr(train_cfg.runner, "save_config", None)
    if not log_dir or not save_config:
        return
    src = Path(__file__).resolve().parents[1] / "envs" / "robots" / "mrobot" / save_config
    if src.exists():
        shutil.copy2(src, os.path.join(log_dir, save_config))


def main():
    os.environ.setdefault("WANDB_MODE", "offline")
    if args_cli.task == "mrobot_dance":
        train_cfg = MrobotMimicDanceLabCfgPPO()
        env_cfg = MrobotMimicDanceIsaacLabEnvCfg()
        env_class = MrobotMimicDanceIsaacLabEnv
        cfg_class = MrobotMimicDanceLabCfg
    else:
        train_cfg = MrobotMimicBPMLabCfgPPO()
        env_cfg = MrobotMimicBPMIsaacLabEnvCfg()
        env_class = MrobotMimicBPMIsaacLabEnv
        cfg_class = MrobotMimicBPMLabCfg
    if args_cli.seed is not None:
        train_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        train_cfg.runner.max_iterations = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        train_cfg.runner.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.run_name is not None:
        train_cfg.runner.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        train_cfg.runner.experiment_name = args_cli.experiment_name
    seed = set_seed(train_cfg.seed)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else cfg_class.env.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.num_steps_per_env = train_cfg.runner.num_steps_per_env
    env_cfg.disable_domain_randomization = args_cli.disable_domain_randomization
    env_cfg.deterministic_reset = args_cli.deterministic_reset
    env_cfg.profile_step_timings = args_cli.profile_step_timings
    env_cfg.profile_step_timing_interval = args_cli.profile_step_timing_interval
    env_cfg.profile_step_timing_warmup = args_cli.profile_step_timing_warmup
    if args_cli.reference_model is not None:
        env_cfg.reference_model_path = args_cli.reference_model
    if args_cli.motion_files is not None:
        env_cfg.motion_files = [item.strip() for item in args_cli.motion_files.split(",") if item.strip()]
    print(
        "[train_mrobot_isaaclab] "
        f"task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, "
        f"action_space={env_cfg.action_space}, observation_space={env_cfg.observation_space}, "
        f"state_space={env_cfg.state_space}, device={env_cfg.sim.device}, "
        f"num_steps_per_env={train_cfg.runner.num_steps_per_env}, "
        f"profile_step_timings={env_cfg.profile_step_timings}",
        flush=True,
    )
    if args_cli.task == "mrobot_dance":
        print(
            "[train_mrobot_isaaclab] motion_files="
            + ", ".join(str(path) for path in (env_cfg.motion_files or cfg_class.motion.files)),
            flush=True,
        )

    direct_env = env_class(env_cfg)
    vec_env = IsaacLabRslRlVecEnv(direct_env)
    vec_env.num_policy_actions = env_cfg.action_space
    cfg_dict = class_to_dict(train_cfg)

    log_dir = None
    if not args_cli.no_log:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)
        os.makedirs(log_dir, exist_ok=True)
        save_training_config(log_dir, train_cfg)

    runner = OnPolicyRunner(vec_env, cfg_dict, log_dir=log_dir, device=args_cli.device)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    vec_env.close()
    if log_dir is not None:
        print("[HumanoidGym-Ex] MRobot IsaacLab PPO log_dir:", log_dir, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
