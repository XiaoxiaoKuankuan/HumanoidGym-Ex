"""MRobot specified-trajectory dance mimic config.

This task is separate from ``mrobot_music``: reference states come from
keypoint ``.npz`` trajectory files, not from the BPM reference network.
"""

import numpy as np

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config import (
    MrobotMimicCommonCfg,
    MrobotMimicCommonCfgPPO,
)
from humanoid_gym_ex.utils.mrobot_trajectory_reference import DEFAULT_DANCE_MOTION_FILES


class MrobotMimicDanceCfg(MrobotMimicCommonCfg):
    class env(MrobotMimicCommonCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        # Dance policy proprio obs excludes waist and base linear velocity:
        # 12 q error + 12 dq + 12 last action + 3 base angular velocity + 3 euler.
        num_single_obs = 42
        num_goal_obs = 19
        num_observations = num_single_obs + num_goal_obs
        single_num_privileged_obs = 45
        num_privileged_obs = 45 + 146 + num_goal_obs
        num_actions = 29
        num_policy_actions = 12
        num_control = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ref_num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        normalize_obs = True

    class motion:
        reference_source = "trajectory"
        files = list(DEFAULT_DANCE_MOTION_FILES)
        allow_legacy_keypoint_fallback = False
        reference_fps = 50
        zero_start_ratio = 0.1
        use_adaptive_phase_sampling = True
        adaptive_bin_size_sec = 1.0
        adaptive_kernel_size = 3
        adaptive_lambda = 0.8
        adaptive_uniform_ratio = 0.1
        foot_contact_height_threshold = 0.08

    class asset(MrobotMimicCommonCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/"
            "Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_guitar.urdf"
        )

    class control(MrobotMimicCommonCfg.control):
        action_scale = 0.25
        use_ref_residual_target = True
        # 50Hz trajectory reference with sim.dt=0.005s -> 4 physics steps per policy step.
        # The Gym/Lab envs still recompute this when match_reference_fps=True.
        decimation = 4
        match_reference_fps = True

    class termination:
        use_tracking_error_termination = True
        waist_z_threshold = 0.25
        waist_ori_threshold = 0.8
        foot_z_threshold = 0.25
        tracking_termination_grace_steps = 5

    class rewards(MrobotMimicCommonCfg.rewards):
        dof_err_w = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        class sigma(MrobotMimicCommonCfg.rewards.sigma):
            foot_height = 0.08
            whole_body_pos = 0.15
            whole_body_rot = 0.2
            whole_body_lin_vel = 1.0
            whole_body_ang_vel = 3.14
            root_height = 0.3
            root_pos = 0.3
            root_rot = 0.4
            root_ang_vel = 1.5
            root_vel = 0.6

        class scales:
            imition_joint_pos = 0.8
            imition_joint_vel = 0.2
            imition_foot_height = 1.0
            imition_root_height = 0.3
            imitation_whole_body_ang_vel = 0.4
            imitation_whole_body_lin_vel = 0.5
            imitation_whole_body_rot = 1.0
            imitation_whole_body_pos = 1.5
            imition_root_pos = 0.3
            imition_root_rot = 0.4
            imitation_root_vel = 0.2
            imition_base_ang_vel = 0.3
            teleop_contact_mask = 1.0
            dof_acc = -5e-6
            action_rate = -0.01
            dof_pos_limits = -2.0
            torque_limits = -1.0

    class noise(MrobotMimicCommonCfg.noise):
        add_noise = True
        noise_level = 1.0

    class normalization(MrobotMimicCommonCfg.normalization):
        class obs_scales(MrobotMimicCommonCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            com_pos = 10.0

        clip_observations = 50
        clip_actions = 50
        actions_filter = True


class MrobotMimicDanceCfgPPO(MrobotMimicCommonCfgPPO):
    seed = 5

    class policy(MrobotMimicCommonCfgPPO.policy):
        init_noise_std = np.array([0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2])
        num_single_obs = 42
        num_goal_obs = 19
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(MrobotMimicCommonCfgPPO.algorithm):
        normalizer_update_iterations = 2400
        entropy_coef = 0.005
        learning_rate = 1e-3
        schedule = "adaptive"
        desired_kl = 0.01
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class runner(MrobotMimicCommonCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 53000
        save_interval = 1000
        experiment_name = "mrobot_dance"
        run_name = ""
        resume = False
        load_run = ""
        checkpoint = -1
        resume_path = None
        save_config = "mrobot_mimic_dance_config.py"
