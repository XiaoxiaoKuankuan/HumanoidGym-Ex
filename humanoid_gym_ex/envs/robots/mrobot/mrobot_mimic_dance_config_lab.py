"""IsaacLab config entry point for MRobot specified-trajectory dance mimic."""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config_lab import (
    MrobotMimicLabCfg,
    MrobotMimicLabCfgPPO,
)
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config import (
    MrobotMimicDanceCfg,
    MrobotMimicDanceCfgPPO,
)


class MrobotMimicDanceLabCfg(MrobotMimicLabCfg):
    lab_joint_effort_limits = MrobotMimicLabCfg.lab_joint_effort_limits
    lab_joint_velocity_limits = MrobotMimicLabCfg.lab_joint_velocity_limits
    lab_joint_position_limits = MrobotMimicLabCfg.lab_joint_position_limits

    class env(MrobotMimicDanceCfg.env):
        num_envs = 4096

    class motion(MrobotMimicDanceCfg.motion):
        pass

    class asset(MrobotMimicDanceCfg.asset):
        pass

    class control(MrobotMimicLabCfg.control):
        action_scale = MrobotMimicDanceCfg.control.action_scale
        use_ref_residual_target = MrobotMimicDanceCfg.control.use_ref_residual_target
        decimation = MrobotMimicDanceCfg.control.decimation

    class domain_rand(MrobotMimicLabCfg.domain_rand):
        pass

    class rewards(MrobotMimicDanceCfg.rewards):
        pass

    class noise(MrobotMimicDanceCfg.noise):
        pass

    class normalization(MrobotMimicDanceCfg.normalization):
        pass


class MrobotMimicDanceLabCfgPPO(MrobotMimicDanceCfgPPO):
    class policy(MrobotMimicDanceCfgPPO.policy):
        pass

    class algorithm(MrobotMimicLabCfgPPO.algorithm):
        pass

    class runner(MrobotMimicDanceCfgPPO.runner):
        save_interval = 500
        experiment_name = "mrobot_dance_isaaclab"
        save_config = "mrobot_mimic_dance_config_lab.py"
        fast_episode_logging = True
