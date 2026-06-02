"""Isaac Lab training config for MRobot BPM mimic (May29_15-05-46_lab snapshot).

Only parameters read by ``isaaclab_env.py`` / ``train_mrobot_isaaclab.py`` are listed.
Disabled features and ankle domain-randomization flags are omitted on purpose.
"""

import numpy as np

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config import (
    ARMATURE_10020_12,
    ARMATURE_6408_25,
    ARMATURE_6416_25,
    ARMATURE_UPPER,
    DAMPING_10020_12,
    DAMPING_6408_25,
    DAMPING_6416_25,
    MrobotMimicCfg,
    MrobotMimicCfgPPO,
    STIFFNESS_10020_12,
    STIFFNESS_6408_25,
    STIFFNESS_6416_25,
)


class MrobotMimicLabCfg(MrobotMimicCfg):
    """Isaac Lab / Isaac Sim entry point (``isaaclab_env`` imports this as ``MrobotMimicCfg``)."""

    lab_joint_effort_limits = {
        "leg_l1_joint": 66.7,
        "leg_l2_joint": 66.7,
        "leg_l3_joint": 46.8,
        "leg_l4_joint": 66.7,
        "leg_l5_joint": 31.5,
        "leg_l6_joint": 31.5,
        "leg_r1_joint": 66.7,
        "leg_r2_joint": 66.7,
        "leg_r3_joint": 46.8,
        "leg_r4_joint": 66.7,
        "leg_r5_joint": 31.5,
        "leg_r6_joint": 31.5,
        "waist_yaw_joint": 31.5,
        "upper_left_1_joint": 75.0,
        "upper_left_2_joint": 75.0,
        "upper_left_3_joint": 36.0,
        "upper_left_4_joint": 75.0,
        "upper_left_5_joint": 36.0,
        "upper_left_6_joint": 36.0,
        "upper_left_7_joint": 36.0,
        "upper_right_1_joint": 75.0,
        "upper_right_2_joint": 75.0,
        "upper_right_3_joint": 36.0,
        "upper_right_4_joint": 75.0,
        "upper_right_5_joint": 36.0,
        "upper_right_6_joint": 36.0,
        "upper_right_7_joint": 36.0,
        "vhead_1_joint": 36.0,
        "vhead_2_joint": 36.0,
    }

    lab_joint_velocity_limits = {
        "leg_l1_joint": 12.6,
        "leg_l2_joint": 12.6,
        "leg_l3_joint": 10.5,
        "leg_l4_joint": 12.6,
        "leg_l5_joint": 12.6,
        "leg_l6_joint": 12.6,
        "leg_r1_joint": 12.6,
        "leg_r2_joint": 12.6,
        "leg_r3_joint": 10.5,
        "leg_r4_joint": 12.6,
        "leg_r5_joint": 12.6,
        "leg_r6_joint": 12.6,
        "waist_yaw_joint": 12.6,
        "upper_left_1_joint": 12.2,
        "upper_left_2_joint": 12.2,
        "upper_left_3_joint": 9.3,
        "upper_left_4_joint": 12.2,
        "upper_left_5_joint": 9.3,
        "upper_left_6_joint": 9.3,
        "upper_left_7_joint": 9.3,
        "upper_right_1_joint": 12.2,
        "upper_right_2_joint": 12.2,
        "upper_right_3_joint": 9.3,
        "upper_right_4_joint": 12.2,
        "upper_right_5_joint": 9.3,
        "upper_right_6_joint": 9.3,
        "upper_right_7_joint": 9.3,
        "vhead_1_joint": 9.3,
        "vhead_2_joint": 9.3,
    }

    lab_joint_position_limits = {
        "leg_l1_joint": [-1.9199, 1.5708],
        "leg_l2_joint": [-0.1745, 1.5708],
        "leg_l3_joint": [-1.5708, 1.5708],
        "leg_l4_joint": [0.0, 2.5307],
        "leg_l5_joint": [-0.8727, 0.5061],
        "leg_l6_joint": [-0.5061, 0.5061],
        "leg_r1_joint": [-1.9199, 1.5708],
        "leg_r2_joint": [-1.5708, 0.1745],
        "leg_r3_joint": [-1.5708, 1.5708],
        "leg_r4_joint": [0.0, 2.5307],
        "leg_r5_joint": [-0.8727, 0.5061],
        "leg_r6_joint": [-0.5061, 0.5061],
        "waist_yaw_joint": [-1.5708, 1.5708],
        "upper_left_1_joint": [-3.1416, 1.0472],
        "upper_left_2_joint": [-0.3491, 3.1416],
        "upper_left_3_joint": [-1.5708, 1.5708],
        "upper_left_4_joint": [-1.8675, 0.0],
        "upper_left_5_joint": [-1.5708, 1.5708],
        "upper_left_6_joint": [-1.0472, 1.0472],
        "upper_left_7_joint": [-1.0472, 1.5708],
        "upper_right_1_joint": [-3.1416, 1.0472],
        "upper_right_2_joint": [-3.1416, 0.3491],
        "upper_right_3_joint": [-1.5708, 1.5708],
        "upper_right_4_joint": [-1.8675, 0.0],
        "upper_right_5_joint": [-1.5708, 1.5708],
        "upper_right_6_joint": [-1.0472, 1.0472],
        "upper_right_7_joint": [-1.5708, 1.0472],
        "vhead_1_joint": [-1.5708, 1.5708],
        "vhead_2_joint": [-0.2618, 0.5236],
    }

    class env(MrobotMimicCfg.env):
        num_single_obs = 45
        num_goal_obs = 19
        num_observations = num_single_obs + num_goal_obs
        num_privileged_obs = 45 + 146 + num_goal_obs
        num_actions = 29
        num_policy_actions = 12
        num_envs = 4096
        episode_length_s = 10
        num_control = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ref_num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        normalize_obs = True

    class motion(MrobotMimicCfg.motion):
        reference_model_path = "BPM_dance/reference_state_keypoint_model.pt"
        bpm_range = [60.0, 170.0]
        include_zero_bpm = True
        sample_integer_bpm = True
        init_phase_range = [0.0, 2.0 * np.pi]
        foot_contact_height_threshold = 0.08

    class safety(MrobotMimicCfg.safety):
        torque_limit = 1

    class asset(MrobotMimicCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/"
            "Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_bass.urdf"
        )
        base_name = "base_link"
        fix_base_link = False
        self_collisions = 0

    class terrain(MrobotMimicCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class noise(MrobotMimicCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(MrobotMimicCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.5
            ang_vel = 0.3
            euler = 0.1

    class init_state(MrobotMimicCfg.init_state):
        pos = [0.0, 0.0, 0.9]
        default_joint_angles = {
            "leg_l1_joint": -0.457,
            "leg_l2_joint": 0.192,
            "leg_l3_joint": 0.062,
            "leg_l4_joint": 0.874,
            "leg_l5_joint": -0.370,
            "leg_l6_joint": -0.126,
            "leg_r1_joint": -0.457,
            "leg_r2_joint": -0.192,
            "leg_r3_joint": -0.062,
            "leg_r4_joint": 0.874,
            "leg_r5_joint": -0.370,
            "leg_r6_joint": 0.126,
            "waist_yaw_joint": 0.0,
            "upper_left_1_joint": 0.171,
            "upper_left_2_joint": 0.327,
            "upper_left_3_joint": 0.442,
            "upper_left_4_joint": -1.093,
            "upper_left_5_joint": 1.257,
            "upper_left_6_joint": -0.058,
            "upper_left_7_joint": -1.514,
            "upper_right_1_joint": -0.149,
            "upper_right_2_joint": -0.755,
            "upper_right_3_joint": 0.306,
            "upper_right_4_joint": -1.865,
            "upper_right_5_joint": 0.259,
            "upper_right_6_joint": 0.156,
            "upper_right_7_joint": 1.073,
            "vhead_1_joint": 0.0,
            "vhead_2_joint": 0.0,
        }

    class control(MrobotMimicCfg.control):
        stiffness = {
            "leg_l1_joint": STIFFNESS_10020_12 / 2,
            "leg_l2_joint": STIFFNESS_10020_12 / 2,
            "leg_l3_joint": STIFFNESS_6416_25 / 2,
            "leg_l4_joint": STIFFNESS_10020_12 / 2,
            "leg_l5_joint": STIFFNESS_6408_25 / 2,
            "leg_l6_joint": STIFFNESS_6408_25 / 2,
            "leg_r1_joint": STIFFNESS_10020_12 / 2,
            "leg_r2_joint": STIFFNESS_10020_12 / 2,
            "leg_r3_joint": STIFFNESS_6416_25 / 2,
            "leg_r4_joint": STIFFNESS_10020_12 / 2,
            "leg_r5_joint": STIFFNESS_6408_25 / 2,
            "leg_r6_joint": STIFFNESS_6408_25 / 2,
            "waist_yaw_joint": 200.0,
            "upper_left_1_joint": 200.0,
            "upper_left_2_joint": 200.0,
            "upper_left_3_joint": 200.0,
            "upper_left_4_joint": 200.0,
            "upper_left_5_joint": 200.0,
            "upper_left_6_joint": 200.0,
            "upper_left_7_joint": 200.0,
            "upper_right_1_joint": 200.0,
            "upper_right_2_joint": 200.0,
            "upper_right_3_joint": 200.0,
            "upper_right_4_joint": 200.0,
            "upper_right_5_joint": 200.0,
            "upper_right_6_joint": 200.0,
            "upper_right_7_joint": 200.0,
            "vhead_1_joint": 200.0,
            "vhead_2_joint": 200.0,
        }
        damping = {
            "leg_l1_joint": DAMPING_10020_12 / 2,
            "leg_l2_joint": DAMPING_10020_12 / 2,
            "leg_l3_joint": DAMPING_6416_25 / 2,
            "leg_l4_joint": DAMPING_10020_12 / 2,
            "leg_l5_joint": DAMPING_6408_25 / 2,
            "leg_l6_joint": DAMPING_6408_25 / 2,
            "leg_r1_joint": DAMPING_10020_12 / 2,
            "leg_r2_joint": DAMPING_10020_12 / 2,
            "leg_r3_joint": DAMPING_6416_25 / 2,
            "leg_r4_joint": DAMPING_10020_12 / 2,
            "leg_r5_joint": DAMPING_6408_25 / 2,
            "leg_r6_joint": DAMPING_6408_25 / 2,
            "waist_yaw_joint": 5.0,
            "upper_left_1_joint": 5.0,
            "upper_left_2_joint": 5.0,
            "upper_left_3_joint": 5.0,
            "upper_left_4_joint": 5.0,
            "upper_left_5_joint": 5.0,
            "upper_left_6_joint": 5.0,
            "upper_left_7_joint": 5.0,
            "upper_right_1_joint": 5.0,
            "upper_right_2_joint": 5.0,
            "upper_right_3_joint": 5.0,
            "upper_right_4_joint": 5.0,
            "upper_right_5_joint": 5.0,
            "upper_right_6_joint": 5.0,
            "upper_right_7_joint": 5.0,
            "vhead_1_joint": 5.0,
            "vhead_2_joint": 5.0,
        }
        action_scale = 0.25
        use_ref_residual_target = False
        decimation = 10

    class sim(MrobotMimicCfg.sim):
        dt = 0.001

        class physx(MrobotMimicCfg.sim.physx):
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23

    class domain_rand(MrobotMimicCfg.domain_rand):
        # Curriculum (Isaac Lab ``update_domain_rand_curriculum``)
        use_curriculum = True
        curriculum_mode = "adaptive"
        curriculum_stage_iters = [0, 700, 1400, 2100]
        push_ratio_schedule = [0.15, 0.35, 0.65, 1.0]
        disturbance_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        restitution_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        pd_ratio_schedule = [0.20, 0.40, 0.70, 1.0]
        motor_strength_ratio_schedule = [0.20, 0.40, 0.70, 1.0]
        delay_ratio_schedule = [0.0, 0.25, 0.50, 1.0]
        payload_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        com_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        link_mass_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        adaptive_min_iteration = 500
        adaptive_metric_ema = 0.90
        adaptive_min_resets = 256
        adaptive_stage_cooldown_iterations = 100
        adaptive_length_ratio_thresholds = [0.0, 0.25, 0.45, 0.70]
        adaptive_fall_ratio_thresholds = [1.0, 0.55, 0.35, 0.20]

        randomize_payload_mass = True
        payload_mass_range = [-4.0, 4.0]
        payload_body_name = "waist_yaw_link"

        randomize_com_displacement = True
        com_body_name = "waist_yaw_link"
        com_offset_x = 0.0
        com_offset_y = 0.0
        com_offset_z = 0.0
        com_x_pos_range = [-0.08, 0.08]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.08, 0.08]

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        static_friction_range = [0.3, 1.6]
        dynamic_friction_range = [0.3, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 0.2]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_default_dof_pos_offset = True
        default_dof_pos_offset_range = [-0.01, 0.01]

        joint_armature_values = [
            ARMATURE_10020_12,
            ARMATURE_10020_12,
            ARMATURE_6416_25,
            ARMATURE_10020_12,
            ARMATURE_6408_25,
            ARMATURE_6408_25,
            ARMATURE_10020_12,
            ARMATURE_10020_12,
            ARMATURE_6416_25,
            ARMATURE_10020_12,
            ARMATURE_6408_25,
            ARMATURE_6408_25,
            ARMATURE_6408_25,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
            ARMATURE_UPPER,
        ]

        disturbance = True
        disturbance_range = [-100.0, 100.0]
        disturbance_s = 8

        push_robots = True
        push_interval_s = [1.0, 3.0]
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.15

        randomize_kp = True
        kp_range = [0.9, 1.1]
        randomize_kd = True
        kd_range = [0.9, 1.1]

        action_delay = True
        action_delay_range = [5, 20]

        randomize_root_xy_reset = False
        root_xy_reset_range = [-0.05, 0.05]
        randomize_root_yaw_reset = True
        root_yaw_reset_range = [-0.17, 0.17]

        randomize_init_dof_pos = True
        init_dof_pos_range = [-0.03, 0.03]

        resample_physx_randomization_on_small_reset = True

    class rewards(MrobotMimicCfg.rewards):
        class sigma(MrobotMimicCfg.rewards.sigma):
            whole_body_pos = 0.15
            whole_body_rot = 0.2
            whole_body_lin_vel = 1.0
            whole_body_ang_vel = 3.14
            root_pos = 0.3
            root_rot = 0.4

        class scales(MrobotMimicCfg.rewards.scales):
            imition_joint_pos = 0.8
            imition_joint_vel = 0.2
            imitation_whole_body_ang_vel = 1.0
            imitation_whole_body_lin_vel = 1.0
            imitation_whole_body_rot = 1.0
            imitation_whole_body_pos = 1.0
            imition_root_pos = 0.5
            imition_root_rot = 0.5
            termination = -50.0
            action_rate = -0.01
            ankle_dof_vel = -1e-3
            dof_pos_limits = -3.0
            torque_limits = -2.0
            ankle_torque_limit = -3.0

    class normalization(MrobotMimicCfg.normalization):
        class obs_scales(MrobotMimicCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            com_pos = 10.0

        clip_actions = 50.0
        actions_filter = True


class MrobotMimicLabCfgPPO(MrobotMimicCfgPPO):
    seed = 5

    class policy(MrobotMimicCfgPPO.policy):
        init_noise_std = np.array(
            [0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2]
        )
        num_single_obs = 45
        num_goal_obs = 19
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(MrobotMimicCfgPPO.algorithm):
        normalizer_update_iterations = 2400
        entropy_coef = 0.005
        learning_rate = 1e-4
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class runner(MrobotMimicCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 53000
        save_interval = 500
        experiment_name = "mrobot_mimic_May_music_BPM_isaaclab"
        save_config = "mrobot_mimic_config_lab.py"
        fast_episode_logging = True
