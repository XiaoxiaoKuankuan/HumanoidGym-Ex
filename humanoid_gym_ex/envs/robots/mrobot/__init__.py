"""MRobot mimic package exports.

Use explicit task names:

- Common: shared base config/env pieces.
- BPM: ``mrobot_music`` BPM/reference-network mimic task.
- Dance: ``mrobot_dance`` specified-trajectory mimic task.
"""

from .mrobot_mimic_common_config import MrobotMimicCommonCfg, MrobotMimicCommonCfgPPO
from .mrobot_mimic_common_config_lab import MrobotMimicCommonLabCfg, MrobotMimicCommonLabCfgPPO
from .mrobot_mimic_bpm_config_gym import MrobotMimicBPMGymCfg, MrobotMimicBPMGymCfgPPO
from .mrobot_mimic_bpm_config_lab import MrobotMimicBPMLabCfg, MrobotMimicBPMLabCfgPPO
from .mrobot_mimic_dance_config_gym import MrobotMimicDanceGymCfg, MrobotMimicDanceGymCfgPPO
from .mrobot_mimic_dance_config_lab import MrobotMimicDanceLabCfg, MrobotMimicDanceLabCfgPPO

try:
    from .mrobot_mimic_common_env import MrobotMimicCommonEnv
    from .mrobot_mimic_bpm_env import MrobotMimicBPMEnv
    from .mrobot_mimic_dance_env import MrobotMimicDanceEnv
except (ModuleNotFoundError, ImportError, OSError):
    MrobotMimicCommonEnv = None
    MrobotMimicBPMEnv = None
    MrobotMimicDanceEnv = None

__all__ = [
    "MrobotMimicCommonCfg",
    "MrobotMimicCommonCfgPPO",
    "MrobotMimicCommonLabCfg",
    "MrobotMimicCommonLabCfgPPO",
    "MrobotMimicBPMGymCfg",
    "MrobotMimicBPMGymCfgPPO",
    "MrobotMimicBPMLabCfg",
    "MrobotMimicBPMLabCfgPPO",
    "MrobotMimicDanceGymCfg",
    "MrobotMimicDanceGymCfgPPO",
    "MrobotMimicDanceLabCfg",
    "MrobotMimicDanceLabCfgPPO",
    "MrobotMimicCommonEnv",
    "MrobotMimicBPMEnv",
    "MrobotMimicDanceEnv",
]
