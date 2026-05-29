from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config import MrobotMimicCfg, MrobotMimicCfgPPO


class MrobotMimicGymCfg(MrobotMimicCfg):
    """IsaacGym training config entry point.

    This class intentionally keeps the original migrated config unchanged.  It
    exists so IsaacGym and IsaacLab can diverge where the simulator APIs require
    different handling without overloading a single file with backend-specific
    switches.
    """


class MrobotMimicGymCfgPPO(MrobotMimicCfgPPO):
    class runner(MrobotMimicCfgPPO.runner):
        save_config = "mrobot_mimic_config_gym.py"
