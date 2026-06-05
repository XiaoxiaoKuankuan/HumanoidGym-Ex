"""IsaacGym config entry point for the MRobot BPM/music mimic task."""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config_gym import (
    MrobotMimicGymCfg as _LegacyMrobotMimicGymCfg,
    MrobotMimicGymCfgPPO as _LegacyMrobotMimicGymCfgPPO,
)


class MrobotMimicBPMGymCfg(_LegacyMrobotMimicGymCfg):
    """BPM-named IsaacGym config.

    It inherits the legacy Gym config without changing behavior; the new class
    name only separates BPM mimic from trajectory dance mimic at import sites.
    """


class MrobotMimicBPMGymCfgPPO(_LegacyMrobotMimicGymCfgPPO):
    class runner(_LegacyMrobotMimicGymCfgPPO.runner):
        save_config = "mrobot_mimic_bpm_config_gym.py"


# Backward-compatible aliases used by existing task registry code and scripts.
MrobotMimicGymCfg = MrobotMimicBPMGymCfg
MrobotMimicGymCfgPPO = MrobotMimicBPMGymCfgPPO

__all__ = [
    "MrobotMimicBPMGymCfg",
    "MrobotMimicBPMGymCfgPPO",
    "MrobotMimicGymCfg",
    "MrobotMimicGymCfgPPO",
]
