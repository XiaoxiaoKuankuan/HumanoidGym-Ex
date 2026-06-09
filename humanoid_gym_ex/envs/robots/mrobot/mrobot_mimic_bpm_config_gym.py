"""IsaacGym config for the MRobot BPM/music mimic task."""

import numpy as np

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config import *  # noqa: F401,F403
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config import (
    MrobotMimicCommonCfg,
    MrobotMimicCommonCfgPPO,
)


class MrobotMimicBPMGymCfg(MrobotMimicCommonCfg):
    """BPM/music mimic task config.

    This is a sibling of the Dance Gym config: both inherit the shared
    ``MrobotMimicCommonCfg`` base, while this class only adds BPM/reference
    network semantics.
    """

    class env(MrobotMimicCommonCfg.env):
        # q(12)+dq(12)+act(12)+angvel(3)+euler(3)+phase_sin/cos(2)+bpm(1)
        num_single_obs = 45
        # ref_dof_pos(12)+ref_dof_vel(12)+waist_z(1)+waist_rp(2)+waist_vel(3)+waist_angvel_z(1)
        num_goal_obs = 31
        num_observations = num_single_obs + num_goal_obs
        single_num_privileged_obs = 45
        num_privileged_obs = 45 + 146 + num_goal_obs
        num_actions = 29
        num_policy_actions = 12
        num_envs = 4096
        episode_length_s = 10
        use_ref_actions = False
        num_aux = 9
        num_control = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ref_num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        normalize_obs = True

    class motion:
        reference_model_path = "BPM_dance/reference_state_keypoint_model.pt"
        bpm_range = [60.0, 170.0]
        include_zero_bpm = True
        sample_integer_bpm = True
        fixed_bpm = None
        randomize_init_phase = True
        init_phase_range = [0.0, 2.0 * np.pi]
        foot_contact_height_threshold = 0.08


class MrobotMimicBPMGymCfgPPO(MrobotMimicCommonCfgPPO):
    class policy(MrobotMimicCommonCfgPPO.policy):
        num_single_obs = 45
        num_goal_obs = 31

    class runner(MrobotMimicCommonCfgPPO.runner):
        experiment_name = "mrobot_mimic_May_music_BPM"
        save_config = "mrobot_mimic_bpm_config_gym.py"


__all__ = ["MrobotMimicBPMGymCfg", "MrobotMimicBPMGymCfgPPO"]
