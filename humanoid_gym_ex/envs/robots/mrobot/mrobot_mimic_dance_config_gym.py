from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config import (
    MrobotMimicDanceCfg,
    MrobotMimicDanceCfgPPO,
)


class MrobotMimicDanceGymCfg(MrobotMimicDanceCfg):
    """IsaacGym entry point for specified-trajectory MRobot dance mimic."""


class MrobotMimicDanceGymCfgPPO(MrobotMimicDanceCfgPPO):
    class runner(MrobotMimicDanceCfgPPO.runner):
        save_config = "mrobot_mimic_dance_config_gym.py"
