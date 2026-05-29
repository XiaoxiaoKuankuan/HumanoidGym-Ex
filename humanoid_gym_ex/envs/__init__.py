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


task_registry = None
_registration_error = None


def register_tasks():
    global task_registry, _registration_error
    global LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
    global LeggedRobot, XBotLCfg, XBotLCfgPPO, XBotLFreeEnv
    global MrobotMimicCfg, MrobotMimicCfgPPO, MrobotMimicEnv
    from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
    from .base.legged_robot import LeggedRobot
    from .robots.humanoid_config import XBotLCfg, XBotLCfgPPO
    from .robots.humanoid_env import XBotLFreeEnv
    from .robots.mrobot.mrobot_mimic_config_gym import MrobotMimicGymCfg, MrobotMimicGymCfgPPO
    MrobotMimicCfg = MrobotMimicGymCfg
    MrobotMimicCfgPPO = MrobotMimicGymCfgPPO
    from .robots.mrobot.mrobot_mimic_env import MrobotMimicEnv
    from humanoid_gym_ex.utils.task_registry import task_registry

    task_registry.register("humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO())
    task_registry.register("mrobot_music", MrobotMimicEnv, MrobotMimicGymCfg(), MrobotMimicGymCfgPPO())
    _registration_error = None
    return task_registry


try:
    register_tasks()
except ModuleNotFoundError as exc:
    if exc.name != "isaacgym":
        raise
    _registration_error = exc
    task_registry = None
except ImportError as exc:
    if "PyTorch was imported before isaacgym" not in str(exc):
        raise
    _registration_error = exc
    task_registry = None
except OSError as exc:
    # IsaacGym's gymtorch extension may try to build in a read-only cache during
    # static CLI usage. Keep config-only imports available; simulator entrypoints
    # will still fail loudly when they actually need IsaacGym.
    _registration_error = exc
    task_registry = None
