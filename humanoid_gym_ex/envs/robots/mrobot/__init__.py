from .mrobot_mimic_common_config import MrobotMimicCommonCfg, MrobotMimicCommonCfgPPO
from .mrobot_mimic_common_config_lab import MrobotMimicCommonLabCfg, MrobotMimicCommonLabCfgPPO
from .mrobot_mimic_bpm_config import (
    MrobotMimicBPMCfg,
    MrobotMimicBPMCfgPPO,
    MrobotMimicCfg,
    MrobotMimicCfgPPO,
)
from .mrobot_mimic_bpm_config_gym import (
    MrobotMimicBPMGymCfg,
    MrobotMimicBPMGymCfgPPO,
    MrobotMimicGymCfg,
    MrobotMimicGymCfgPPO,
)
from .mrobot_mimic_bpm_config_lab import (
    MrobotMimicBPMLabCfg,
    MrobotMimicBPMLabCfgPPO,
    MrobotMimicLabCfg,
    MrobotMimicLabCfgPPO,
)
from .mrobot_mimic_dance_config import MrobotMimicDanceCfg, MrobotMimicDanceCfgPPO
from .mrobot_mimic_dance_config_gym import MrobotMimicDanceGymCfg, MrobotMimicDanceGymCfgPPO
from .mrobot_mimic_dance_config_lab import MrobotMimicDanceLabCfg, MrobotMimicDanceLabCfgPPO

try:
    from .mrobot_mimic_common_env import MrobotMimicCommonEnv
    from .mrobot_mimic_bpm_env import MrobotMimicBPMEnv, MrobotMimicEnv
    from .mrobot_mimic_dance_env import MrobotMimicDanceEnv
except (ModuleNotFoundError, ImportError, OSError):
    MrobotMimicCommonEnv = None
    MrobotMimicBPMEnv = None
    MrobotMimicEnv = None
    MrobotMimicDanceEnv = None

__all__ = [
    "MrobotMimicCommonCfg",
    "MrobotMimicCommonCfgPPO",
    "MrobotMimicCommonLabCfg",
    "MrobotMimicCommonLabCfgPPO",
    "MrobotMimicBPMCfg",
    "MrobotMimicBPMCfgPPO",
    "MrobotMimicCfg",
    "MrobotMimicCfgPPO",
    "MrobotMimicBPMGymCfg",
    "MrobotMimicBPMGymCfgPPO",
    "MrobotMimicGymCfg",
    "MrobotMimicGymCfgPPO",
    "MrobotMimicBPMLabCfg",
    "MrobotMimicBPMLabCfgPPO",
    "MrobotMimicLabCfg",
    "MrobotMimicLabCfgPPO",
    "MrobotMimicDanceCfg",
    "MrobotMimicDanceCfgPPO",
    "MrobotMimicDanceGymCfg",
    "MrobotMimicDanceGymCfgPPO",
    "MrobotMimicDanceLabCfg",
    "MrobotMimicDanceLabCfgPPO",
    "MrobotMimicCommonEnv",
    "MrobotMimicBPMEnv",
    "MrobotMimicEnv",
    "MrobotMimicDanceEnv",
]
