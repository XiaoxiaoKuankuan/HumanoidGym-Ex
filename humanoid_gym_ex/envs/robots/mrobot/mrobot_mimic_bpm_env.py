"""IsaacGym env entry point for the MRobot BPM/music mimic task."""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_env import MrobotMimicCommonEnv


class MrobotMimicBPMEnv(MrobotMimicCommonEnv):
    """BPM/music mimic env.

    This class is intentionally thin: the migrated common env still contains
    the BPM reference-network implementation, while Dance overrides the parts
    that load and advance trajectory references.
    """

# Backward-compatible alias.
MrobotMimicEnv = MrobotMimicBPMEnv

__all__ = ["MrobotMimicBPMEnv", "MrobotMimicEnv"]
