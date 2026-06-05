"""Shared IsaacGym env base for MRobot mimic tasks.

The implementation is still provided by the migrated ``mrobot_mimic_env.py``
class, which contains the common action, observation, reward, and reference
buffer helpers used by both BPM and trajectory-dance tasks.
"""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_env import (
    MrobotMimicEnv as MrobotMimicCommonEnv,
)

__all__ = ["MrobotMimicCommonEnv"]
