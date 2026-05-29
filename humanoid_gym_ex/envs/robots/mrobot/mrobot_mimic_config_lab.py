from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config import MrobotMimicCfg, MrobotMimicCfgPPO


class MrobotMimicLabCfg(MrobotMimicCfg):
    """IsaacLab/IsaacSim training config entry point."""

    # Explicit IsaacLab actuator limits copied from
    # resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/urdf/
    # CASBOT02_ENCOS_7dof_shell_20251015_bass.urdf.
    # Keeping them here makes it clear what PhysX actuator limits the Lab
    # workflow should use, instead of relying on importer defaults.
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

    class domain_rand(MrobotMimicCfg.domain_rand):
        # IsaacLab exposes static and dynamic friction separately.  The legacy
        # IsaacGym config had one friction_range, so these default to matching
        # ranges and can be tuned independently from this file.
        static_friction_range = [0.3, 1.6]
        dynamic_friction_range = [0.3, 1.2]

        # IsaacLab writes the fixed armature values to PhysX even when
        # randomize_joint_armature=False, matching the IsaacGym _process_dof_props
        # behavior.  Set this True to sample joint_armature_range per reset.
        randomize_joint_armature = False

        # Match the IsaacGym mimic task semantics: every environment episode
        # reset re-samples material/mass/COM/PhysX joint randomization.
        # This is more expensive in IsaacLab because these writes pass through
        # root_physx_view CPU tensor setters, but it keeps the randomization
        # distribution aligned with the Gym task.
        resample_physx_randomization_on_small_reset = True

        # Lab training uses the shared full-joint randomization only.  Keep all
        # ankle-specific randomization disabled so IsaacLab and IsaacGym differ
        # less in ankle sensing/PD behavior while debugging the simulator gap.
        randomize_ankle_pd = False
        randomize_ankle_motor_offset = False
        randomize_ankle_obs_pos_bias = False
        randomize_ankle_obs_vel_bias = False
        randomize_ankle_obs_vel_noise = False
        randomize_ankle_obs_vel_delay = False
        randomize_ankle_obs_vel_filter = False
        randomize_ankle_pd_dq_noise = False
        randomize_ankle_pd_dq_delay = False
        randomize_ankle_pd_dq_filter = False
        default_dof_pos_offset_ankle_range = None


class MrobotMimicLabCfgPPO(MrobotMimicCfgPPO):
    class runner(MrobotMimicCfgPPO.runner):
        experiment_name = "mrobot_mimic_May_music_BPM_isaaclab"
        save_config = "mrobot_mimic_config_lab.py"
        # MRobot IsaacLab 默认 4096 env，episode 较长。旧的逐步
        # nonzero+cpu 日志会在 rollout 内反复同步 GPU；这里只对
        # MRobot Lab 训练启用每轮一次的快速 episode 均值统计。
        fast_episode_logging = True
