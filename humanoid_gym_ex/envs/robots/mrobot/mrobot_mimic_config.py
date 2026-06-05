"""Backward-compatible import path for the MRobot BPM mimic config.

New code should import one of:

- ``mrobot_mimic_common_config.py`` for shared MRobot mimic defaults.
- ``mrobot_mimic_bpm_config.py`` for the BPM/music task.
- ``mrobot_mimic_dance_config.py`` for the specified-trajectory dance task.
"""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_bpm_config import *  # noqa: F401,F403
