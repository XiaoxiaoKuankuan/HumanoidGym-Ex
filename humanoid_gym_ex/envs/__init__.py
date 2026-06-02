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
    """Register Isaac Gym tasks. Call from train/play after isaacgym is imported."""
    global task_registry, _registration_error
    global LeggedRobot, XBotLCfg, XBotLCfgPPO, XBotLFreeEnv
    global MrobotMimicCfg, MrobotMimicCfgPPO, MrobotMimicEnv
    global MrobotMimicDanceGymCfg, MrobotMimicDanceGymCfgPPO, MrobotMimicDanceEnv
    from .base.legged_robot import LeggedRobot
    from .robots.humanoid_config import XBotLCfg, XBotLCfgPPO
    from .robots.humanoid_env import XBotLFreeEnv
    from .robots.mrobot.mrobot_mimic_config_gym import MrobotMimicGymCfg, MrobotMimicGymCfgPPO
    MrobotMimicCfg = MrobotMimicGymCfg
    MrobotMimicCfgPPO = MrobotMimicGymCfgPPO
    from .robots.mrobot.mrobot_mimic_env import MrobotMimicEnv
    from .robots.mrobot.mrobot_mimic_dance_config_gym import MrobotMimicDanceGymCfg, MrobotMimicDanceGymCfgPPO
    from .robots.mrobot.mrobot_mimic_dance_env import MrobotMimicDanceEnv
    from humanoid_gym_ex.utils.task_registry import task_registry as registry

    registry.register("humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO())
    registry.register("mrobot_music", MrobotMimicEnv, MrobotMimicGymCfg(), MrobotMimicGymCfgPPO())
    registry.register("mrobot_dance", MrobotMimicDanceEnv, MrobotMimicDanceGymCfg(), MrobotMimicDanceGymCfgPPO())
    task_registry = registry
    _registration_error = None
    return registry
