"""IsaacLab/IsaacSim config entry point for the MRobot BPM/music mimic task."""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config_lab import (
    MrobotMimicLabCfg as _LegacyMrobotMimicLabCfg,
    MrobotMimicLabCfgPPO as _LegacyMrobotMimicLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config_lab import (
    MrobotMimicCommonLabCfg,
    MrobotMimicCommonLabCfgPPO,
)


class MrobotMimicBPMLabCfg(MrobotMimicCommonLabCfg):
    """BPM-named IsaacLab config.

    The legacy Lab config remains for compatibility, but the public BPM class
    now inherits the same common Lab base as the Dance Lab config.
    """

    lab_joint_effort_limits = _LegacyMrobotMimicLabCfg.lab_joint_effort_limits
    lab_joint_velocity_limits = _LegacyMrobotMimicLabCfg.lab_joint_velocity_limits
    lab_joint_position_limits = _LegacyMrobotMimicLabCfg.lab_joint_position_limits

    class env(_LegacyMrobotMimicLabCfg.env):
        pass

    class motion(_LegacyMrobotMimicLabCfg.motion):
        pass

    class safety(_LegacyMrobotMimicLabCfg.safety):
        pass

    class asset(_LegacyMrobotMimicLabCfg.asset):
        pass

    class terrain(_LegacyMrobotMimicLabCfg.terrain):
        pass

    class noise(_LegacyMrobotMimicLabCfg.noise):
        class noise_scales(_LegacyMrobotMimicLabCfg.noise.noise_scales):
            pass

    class init_state(_LegacyMrobotMimicLabCfg.init_state):
        pass

    class control(_LegacyMrobotMimicLabCfg.control):
        pass

    class sim(_LegacyMrobotMimicLabCfg.sim):
        class physx(_LegacyMrobotMimicLabCfg.sim.physx):
            pass

    class domain_rand(_LegacyMrobotMimicLabCfg.domain_rand):
        pass

    class rewards(_LegacyMrobotMimicLabCfg.rewards):
        class sigma(_LegacyMrobotMimicLabCfg.rewards.sigma):
            pass

        class scales(_LegacyMrobotMimicLabCfg.rewards.scales):
            pass

    class normalization(_LegacyMrobotMimicLabCfg.normalization):
        class obs_scales(_LegacyMrobotMimicLabCfg.normalization.obs_scales):
            pass


class MrobotMimicBPMLabCfgPPO(MrobotMimicCommonLabCfgPPO):
    class policy(_LegacyMrobotMimicLabCfgPPO.policy):
        pass

    class algorithm(_LegacyMrobotMimicLabCfgPPO.algorithm):
        pass

    class runner(_LegacyMrobotMimicLabCfgPPO.runner):
        save_config = "mrobot_mimic_bpm_config_lab.py"


# Backward-compatible aliases used by existing scripts.
MrobotMimicLabCfg = MrobotMimicBPMLabCfg
MrobotMimicLabCfgPPO = MrobotMimicBPMLabCfgPPO

__all__ = [
    "MrobotMimicBPMLabCfg",
    "MrobotMimicBPMLabCfgPPO",
    "MrobotMimicLabCfg",
    "MrobotMimicLabCfgPPO",
]
