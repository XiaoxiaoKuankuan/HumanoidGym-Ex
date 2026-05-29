# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import time
import torch
import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic
from .normalizer import EmpiricalNormalization
from humanoid_gym_ex.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        self.num_policy_actions = int(getattr(self.env, "num_policy_actions", self.env.num_actions))
        self.num_aux = int(getattr(self.env, "num_aux", getattr(getattr(self.env, "cfg", None).env, "num_aux", 0) if hasattr(getattr(self.env, "cfg", None), "env") else 0))
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.num_policy_actions, num_aux=self.num_aux, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        alg_kwargs = {key: value for key, value in self.alg_cfg.items() if key != "normalizer_update_iterations"}
        self.alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.fast_episode_logging = bool(self.cfg.get("fast_episode_logging", False))

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.num_policy_actions],
            [self.num_aux] if self.num_aux > 0 else None,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.normalize_obs = bool(getattr(getattr(self.env, "cfg", None).env, "normalize_obs", False) if hasattr(getattr(self.env, "cfg", None), "env") else False)
        self.normalizer_update_iterations = int(self.alg_cfg.get("normalizer_update_iterations", 5000))
        if self.normalize_obs:
            share_normalizer = self.env.num_privileged_obs is None or self.env.num_obs == self.env.num_privileged_obs
            self.obs_normalizer = EmpiricalNormalization(shape=self.env.num_obs).to(self.device)
            self.critic_obs_normalizer = (
                self.obs_normalizer
                if share_normalizer
                else EmpiricalNormalization(shape=self.env.num_privileged_obs).to(self.device)
            )
        else:
            self.obs_normalizer = None
            self.critic_obs_normalizer = None

        _, _ = self.env.reset()
        if hasattr(self.env, "update_domain_rand_curriculum"):
            self.env.update_domain_rand_curriculum(self.current_learning_iteration, force=True)

    def _normalize_observations(self, obs, critic_obs, iteration=None):
        if not self.normalize_obs:
            return obs, critic_obs
        if iteration is not None and iteration < self.normalizer_update_iterations:
            self.obs_normalizer.update(obs)
            if self.critic_obs_normalizer is not self.obs_normalizer:
                self.critic_obs_normalizer.update(critic_obs)
        return self.obs_normalizer.act(obs), self.critic_obs_normalizer.act(critic_obs)

    @staticmethod
    def _unpack_step(step_result):
        if len(step_result) == 6:
            obs, privileged_obs, rewards, dones, infos, aux = step_result
            if infos is None:
                infos = {}
            infos["aux"] = aux
            return obs, privileged_obs, rewards, dones, infos
        return step_result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            wandb.init(
                project="XBot",
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg,
            )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs, critic_obs = self._normalize_observations(obs, critic_obs)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        fast_episode_count = torch.zeros((), dtype=torch.float, device=self.device)
        fast_episode_reward_sum = torch.zeros((), dtype=torch.float, device=self.device)
        fast_episode_length_sum = torch.zeros((), dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.env, "update_domain_rand_curriculum"):
                self.env.update_domain_rand_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self._unpack_step(self.env.step(actions))
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    obs, critic_obs = self._normalize_observations(obs, critic_obs, it)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards.view(-1)
                        cur_episode_length += 1
                        if self.fast_episode_logging:
                            done_mask = (dones > 0).view(-1)
                            done_float = done_mask.float()
                            fast_episode_count += done_float.sum()
                            fast_episode_reward_sum += torch.sum(cur_reward_sum * done_float)
                            fast_episode_length_sum += torch.sum(cur_episode_length * done_float)
                            keep_mask = (~done_mask).float()
                            cur_reward_sum *= keep_mask
                            cur_episode_length *= keep_mask
                        else:
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(
                                cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            lenbuffer.extend(
                                cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                if self.log_dir is not None and self.fast_episode_logging:
                    fast_stats = torch.stack(
                        (fast_episode_count, fast_episode_reward_sum, fast_episode_length_sum)
                    ).detach().cpu()
                    episode_count = int(fast_stats[0].item())
                    if episode_count > 0:
                        mean_episode_reward = float(fast_stats[1].item()) / max(float(episode_count), 1.0)
                        mean_episode_length = float(fast_stats[2].item()) / max(float(episode_count), 1.0)
                        repeat = min(episode_count, rewbuffer.maxlen or episode_count)
                        rewbuffer.extend([mean_episode_reward] * repeat)
                        lenbuffer.extend([mean_episode_length] * repeat)
                    fast_episode_count.zero_()
                    fast_episode_reward_sum.zero_()
                    fast_episode_length_sum.zero_()

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            update_result = self.alg.update()
            mean_value_loss, mean_surrogate_loss = update_result[:2]
            mean_aux_loss = update_result[2] if len(update_result) > 2 else 0.0
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
                )
            )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            info_keys = list(locs["ep_infos"][0].keys())
            for ep_info in locs["ep_infos"][1:]:
                for key in ep_info.keys():
                    if key not in info_keys:
                        info_keys.append(key)
            for key in info_keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/aux_loss", locs.get("mean_aux_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Aux loss:':>{pad}} {locs.get('mean_aux_loss', 0.0):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Aux loss:':>{pad}} {locs.get('mean_aux_loss', 0.0):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "obs_normalizer": self.obs_normalizer.state_dict() if self.obs_normalizer is not None else None,
                "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.critic_obs_normalizer is not None else None,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self.obs_normalizer is not None and loaded_dict.get("obs_normalizer") is not None:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer"])
        if self.critic_obs_normalizer is not None and loaded_dict.get("critic_obs_normalizer") is not None:
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(device)

        if self.obs_normalizer is None:
            return self.alg.actor_critic.act_inference

        def policy(obs):
            return self.alg.actor_critic.act_inference(self.obs_normalizer.act(obs))

        return policy

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
