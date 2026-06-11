"""IsaacLab config entry point for MRobot specified-trajectory dance mimic."""

import numpy as np

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config_lab import (
    MrobotMimicCommonLabCfg,
    MrobotMimicCommonLabCfgPPO,
)
from humanoid_gym_ex.utils.mrobot_trajectory_reference import DEFAULT_DANCE_MOTION_FILES
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config import (
    ARMATURE_10020_12,
    ARMATURE_6408_25,
    ARMATURE_6416_25,
    ARMATURE_UPPER,
    DAMPING_10020_12,
    DAMPING_6408_25,
    DAMPING_6416_25,
    STIFFNESS_10020_12,
    STIFFNESS_6408_25,
    STIFFNESS_6416_25,
)


class MrobotMimicDanceLabCfg(MrobotMimicCommonLabCfg):
    lab_joint_effort_limits = MrobotMimicCommonLabCfg.lab_joint_effort_limits
    lab_joint_velocity_limits = MrobotMimicCommonLabCfg.lab_joint_velocity_limits
    lab_joint_position_limits = MrobotMimicCommonLabCfg.lab_joint_position_limits

    class sim(MrobotMimicCommonLabCfg.sim):
        dt = 0.005  # 200 Hz low-level physics

        class physx(MrobotMimicCommonLabCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 4
            max_depenetration_velocity = 1.0

    class env(MrobotMimicCommonLabCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        num_single_obs = 42
        # ref_dof_pos(12)+ref_dof_vel(12)+waist_z(1)+waist_rp(2)
        # +waist_vel(3)+waist_angvel_z(1)+feet_contact(2), where
        # feet_contact uses 1=contact, 0=swing.
        num_goal_obs = 33
        num_observations = num_single_obs + num_goal_obs
        single_num_privileged_obs = 45
        num_privileged_obs = 45 + 119 + num_goal_obs
        num_actions = 29
        num_policy_actions = 12
        num_envs = 4096
        episode_length_s = 10
        num_control = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ref_num_notcontrol = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        normalize_obs = True

    class motion:
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

    class asset(MrobotMimicCommonLabCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/"
            "Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_guitar.urdf"
        )
        base_name = "base_link"
        waist_name = "waist_yaw_link"
        foot_name = MrobotMimicCommonLabCfg.asset.foot_name
        ankle_name = MrobotMimicCommonLabCfg.asset.ankle_name
        knee_name = MrobotMimicCommonLabCfg.asset.knee_name
        hip_name = MrobotMimicCommonLabCfg.asset.hip_name
        pelvic_yaw_name = MrobotMimicCommonLabCfg.asset.pelvic_yaw_name
        head_name = MrobotMimicCommonLabCfg.asset.head_name
        tracking_body_names = [
            "waist_yaw_link",
            "left_leg_pelvic_roll_link",
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_pelvic_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
        ]
        terminate_after_contacts_on = list(MrobotMimicCommonLabCfg.asset.terminate_after_contacts_on)
        penalize_contacts_on = list(MrobotMimicCommonLabCfg.asset.penalize_contacts_on)
        fix_base_link = False
        self_collisions = 0

    class terrain(MrobotMimicCommonLabCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class init_state(MrobotMimicCommonLabCfg.init_state):
        pos = [0.0, 0.0, 0.9]
        default_joint_angles = {
            "leg_l1_joint": -0.185,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.36,
            "leg_l5_joint": -0.175,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": -0.185,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.36,
            "leg_r5_joint": -0.175,
            "leg_r6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "upper_left_1_joint": 0.0,
            "upper_left_2_joint": 0.0,
            "upper_left_3_joint": 0.0,
            "upper_left_4_joint": 0.0,
            "upper_left_5_joint": 0.0,
            "upper_left_6_joint": 0.0,
            "upper_left_7_joint": 0.0,
            "upper_right_1_joint": 0.0,
            "upper_right_2_joint": 0.0,
            "upper_right_3_joint": 0.0,
            "upper_right_4_joint": 0.0,
            "upper_right_5_joint": 0.0,
            "upper_right_6_joint": 0.0,
            "upper_right_7_joint": 0.0,
            "vhead_1_joint": 0.0,
            "vhead_2_joint": 0.0,
        }

    class control(MrobotMimicCommonLabCfg.control):
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
        armature = {
            "leg_l1_joint": ARMATURE_10020_12,
            "leg_l2_joint": ARMATURE_10020_12,
            "leg_l3_joint": ARMATURE_6416_25,
            "leg_l4_joint": ARMATURE_10020_12,
            "leg_l5_joint": ARMATURE_6408_25,
            "leg_l6_joint": ARMATURE_6408_25,
            "leg_r1_joint": ARMATURE_10020_12,
            "leg_r2_joint": ARMATURE_10020_12,
            "leg_r3_joint": ARMATURE_6416_25,
            "leg_r4_joint": ARMATURE_10020_12,
            "leg_r5_joint": ARMATURE_6408_25,
            "leg_r6_joint": ARMATURE_6408_25,
            "waist_yaw_joint": ARMATURE_6408_25,
            "upper_left_1_joint": ARMATURE_UPPER,
            "upper_left_2_joint": ARMATURE_UPPER,
            "upper_left_3_joint": ARMATURE_UPPER,
            "upper_left_4_joint": ARMATURE_UPPER,
            "upper_left_5_joint": ARMATURE_UPPER,
            "upper_left_6_joint": ARMATURE_UPPER,
            "upper_left_7_joint": ARMATURE_UPPER,
            "upper_right_1_joint": ARMATURE_UPPER,
            "upper_right_2_joint": ARMATURE_UPPER,
            "upper_right_3_joint": ARMATURE_UPPER,
            "upper_right_4_joint": ARMATURE_UPPER,
            "upper_right_5_joint": ARMATURE_UPPER,
            "upper_right_6_joint": ARMATURE_UPPER,
            "upper_right_7_joint": ARMATURE_UPPER,
            "vhead_1_joint": ARMATURE_UPPER,
            "vhead_2_joint": ARMATURE_UPPER,
        }
        action_scale = 0.25
        ankle_action_scale = 0.2
        ankle_action_scale_indices = [4, 5, 10, 11]
        use_ref_residual_target = False
        decimation = 4  # 50 Hz policy/control with sim.dt=0.005
        match_reference_fps = True

    class termination:
        use_tracking_error_termination = True
        waist_z_threshold = 0.25
        waist_ori_threshold = 0.8
        foot_z_threshold = 0.25  # 0.25
        tracking_termination_grace_steps = 5

    class domain_rand(MrobotMimicCommonLabCfg.domain_rand):
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
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.95, 1.05]

        randomize_friction = True
        static_friction_range = [0.3, 1.6]
        dynamic_friction_range = [0.3, 1.4]

        randomize_restitution = True
        restitution_range = [0.0, 0.5]

        randomize_motor_strength = False
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

        disturbance = False
        disturbance_range = [-100.0, 100.0]
        disturbance_s = 8

        push_robots = True
        push_interval_s = [1.0, 3.0]
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.15

        randomize_kp = False
        kp_range = [0.9, 1.1]
        randomize_kd = False
        kd_range = [0.9, 1.1]

        action_delay = False
        action_delay_range = [0, 0]

        randomize_root_xy_reset = True
        root_xy_reset_range = [-0.05, 0.05]
        randomize_root_yaw_reset = True
        root_yaw_reset_range = [-0.17, 0.17]

        randomize_init_dof_pos = True
        init_dof_pos_range = [-0.03, 0.03]

        # Lab-only speed knob: resetting PhysX materials/mass/COM/armature on
        # every small reset is expensive.  Runtime buffers are still resampled
        # every episode; PhysX properties are rewritten on full-env reset.
        resample_physx_randomization_on_small_reset = False

    class rewards(MrobotMimicCommonLabCfg.rewards):
        # Only hide duplicate weighted tracking reward components from console/TensorBoard
        # episode logs. Reward computation still uses the scales below, and raw
        # tracking quality remains logged as score_*.
        hide_tracking_reward_logs = True
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

        class sigma(MrobotMimicCommonLabCfg.rewards.sigma):
            foot_height = 0.08
            whole_body_pos = 0.15
            whole_body_rot = 0.3
            whole_body_lin_vel = 1.0
            whole_body_ang_vel = 3.14
            root_height = 0.3
            root_pos = 0.3
            root_rot = 0.4
            root_ang_vel = 1.5
            root_vel = 0.6

        class scales:
            imition_joint_pos = 0.2
            imition_joint_vel = 0.1
            # imition_foot_height = 0.5
            # imition_root_height = 0.3
            imitation_whole_body_ang_vel = 1.0
            imitation_whole_body_lin_vel = 1.0
            imitation_whole_body_rot = 1.0
            imitation_whole_body_pos = 1.5
            imition_root_pos = 0.5
            imition_root_rot = 0.5
            # imitation_root_vel = 0.2
            # imition_base_ang_vel = 0.3
            teleop_contact_mask = 0.5
            # foot_slip = -1
            # pre_landing_foot_z_vel = -0.08
            # feet_contact_forces = -0.02
            # dof_acc = -5e-6
            action_rate = -0.1
            dof_pos_limits = -8.0
            # torque_limits = 0.0
            # ankle_torque_limit = 0.0
            termination = -10.0  

    class noise(MrobotMimicCommonLabCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(MrobotMimicCommonLabCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.5
            ang_vel = 0.3
            euler = 0.1

    class normalization(MrobotMimicCommonLabCfg.normalization):
        class obs_scales(MrobotMimicCommonLabCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            com_pos = 10.0

        clip_observations = 50
        clip_actions = 50
        actions_filter = False


class MrobotMimicDanceLabCfgPPO(MrobotMimicCommonLabCfgPPO):
    seed = 5

    class policy(MrobotMimicCommonLabCfgPPO.policy):
        init_noise_std = np.array([0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2])
        num_single_obs = 42
        num_goal_obs = 33
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(MrobotMimicCommonLabCfgPPO.algorithm):
        normalizer_update_iterations = 2400
        entropy_coef = 0.005
        learning_rate = 1e-3
        schedule = "adaptive"
        desired_kl = 0.01
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class runner(MrobotMimicCommonLabCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 83000
        save_interval = 1000
        experiment_name = "mrobot_dance_isaaclab"
        run_name = ""
        resume = False
        load_run = ""
        checkpoint = -1
        resume_path = None
        save_config = "mrobot_mimic_dance_config_lab.py"
        fast_episode_logging = True
