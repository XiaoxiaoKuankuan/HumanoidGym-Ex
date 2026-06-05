# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid_gym_ex.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

# 腿部和腰部电机惯量 / 刚度 / 阻尼（仅用于腿、腰关节 PD）
# 大关节 1,2,4: EC-A10020-P1-12
ARMATURE_10020_12 = 0.07
# 小关节 3,5,6 和腰: EC-A6408-P2-25
ARMATURE_6408_25 = 0.039
# 6416-25
ARMATURE_6416_25 = 0.065
ARMATURE_6416_25_RANGE = [0.063, 0.07]


NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_10020_12 = ARMATURE_10020_12 * NATURAL_FREQ**2
STIFFNESS_6408_25 = ARMATURE_6408_25 * NATURAL_FREQ**2

DAMPING_10020_12 = 2.0 * DAMPING_RATIO * ARMATURE_10020_12 * NATURAL_FREQ
DAMPING_6408_25 = 2.0 * DAMPING_RATIO * ARMATURE_6408_25 * NATURAL_FREQ

# 乐队3关节换成了6416-25
STIFFNESS_6416_25 = ARMATURE_6416_25 * NATURAL_FREQ**2  # 256.6097056
DAMPING_6416_25 = 2.0 * DAMPING_RATIO * ARMATURE_6416_25 * NATURAL_FREQ  # 16.33628152

# 域随机：armature 采样范围（与上面电机类型对应）
ARMATURE_10020_12_RANGE = [0.068, 0.075]  # 0.07
ARMATURE_6408_25_RANGE = [0.037, 0.042]  # 0.039
ARMATURE_UPPER_RANGE = [0.0, 0.01]
ARMATURE_UPPER = 0.005

'''
STIFFNESS_10020_12 = 276.348923229
STIFFNESS_6408_25 = 153.965828656

DAMPING_10020_12 = 17.5929188596 / 2
DAMPING_6408_25 = 9.80176907892 / 2
'''

# 2号从10020-12换成了10020-24：
ARMATURE_10020_24 = 0.2747
# 3号从6408换成了8116：
ARMATURE_8116 = 0.063828

STIFFNESS_10020_24 = ARMATURE_10020_24 * NATURAL_FREQ**2
STIFFNESS_8116 = ARMATURE_8116 * NATURAL_FREQ**2

DAMPING_10020_24 = 2.0 * DAMPING_RATIO * ARMATURE_10020_24 * NATURAL_FREQ
DAMPING_8116 = 2.0 * DAMPING_RATIO * ARMATURE_8116 * NATURAL_FREQ


ARMATURE_10020_24_RANGE = [0.25, 0.30]
ARMATURE_8116_RANGE = [0.058, 0.068]

'''
STIFFNESS_10020_24 = 1084.67738
STIFFNESS_8116 = 251.9828

DAMPING_10020_24 = 69.0396 / 2
DAMPING_8116 = 16.04172579 / 2
'''


class MrobotMimicCommonCfg(LeggedRobotCfg):
    """Shared MRobot mimic defaults used by both BPM and Dance tasks."""

    class env(LeggedRobotCfg.env):
        # Common default: use the current proprio/goal frame only.  Task
        # configs override the exact proprio terms and observation dimensions.
        frame_stack = 1
        c_frame_stack = 1
        d_frame_stack = 10
        # Actor 只控制腿部 12 个关节，不再控制腰。
        num_single_obs = 45
        # ref_dof_pos(12)+waist_z(1)+waist_rp(2)+waist_vel(3)+waist_angvel_z(1)
        num_goal_obs = 19
        num_observations = num_single_obs + num_goal_obs
        single_num_privileged_obs = 45  # height(1)+roll_pitch(2)+dof_pos(12)+dof_vel(12)+act(12)+linvel(3)+angvel(3)
        num_privileged_obs = 45 + 146 + num_goal_obs  # 当前 priv_hist_part + priv_curr + 当前 goal
        single_num_disc_obs = 67
        num_disc_obs = int(d_frame_stack * single_num_disc_obs)
        num_actions = 29  # 28  29
        num_policy_actions = 12
        num_envs = 4096
        episode_length_s = 10  # episode length in seconds  10
        use_ref_actions = False
        num_aux = 9
        # 只控制双腿 12 个关节；腰关节 index=12 进入非受控参考跟随。
        num_control = [0,1,2,3,4,5,6,7,8,9,10,11]
        num_notcontrol = [12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        ref_num_notcontrol = [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        rand_init_coef = 0.8
        normalize_obs = True  # 是否对观测做 running mean/std 归一化  False
        
    class motion:
        reference_source = "common"
        foot_contact_height_threshold = 0.08



    class safety:
        # safety factors
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_guitar.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_bass.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/casbot02_large_torq/CB02_URDF_ECS_130.urdf'  # 换了2，3关节的urdf
        name = "L1"
        head_name = "head_yaw_link"
        foot_name = "ankle_roll_link"
        ankle_name = "ankle_pitch_link"
        knee_name = "knee_pitch_link"
        hip_name = 'pelvic_roll_link'  # 'pelvic_yaw_link'  'pelvic_roll_link'
        pelvic_yaw_name = 'pelvic_yaw_link'
        base_name = 'base_link'
        waist_name = 'waist_yaw_link'
        

        terminate_after_contacts_on = [
            'base_link', 
            'waist_yaw_link',
            'left_leg_pelvic_yaw_link', 
            'left_leg_knee_pitch_link', 
            # 'left_upbody_link_2',
            # 'left_upbody_link_3',
            # 'left_upbody_link_4',

            'right_leg_pelvic_yaw_link', 
            'right_leg_knee_pitch_link',
            # 'right_upbody_link_2',
            # 'right_upbody_link_3',
            # 'right_upbody_link_4',
        ]

        #terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = [
            'base_link', 
            'waist_yaw_link',
            'left_leg_pelvic_yaw_link', 
            'left_leg_knee_pitch_link', 
            # 'left_upbody_link_2',
            # 'left_upbody_link_3',
            # 'left_upbody_link_4',

            'right_leg_pelvic_yaw_link', 
            'right_leg_knee_pitch_link', 
            # 'right_upbody_link_2',
            # 'right_upbody_link_3',
            # 'right_upbody_link_4',
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.4, 0.0, 0.6, 0.0, 0.0, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1.    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            ang_vel = 0.3
            lin_vel = 0.5
            quat = 0.03
            gravity = 0.05
            height_measurements = 0.1
            euler = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.9]

        default_joint_angles = {
            # 'leg_l1_joint': -0.185,
            # 'leg_l2_joint': 0.0,
            # 'leg_l3_joint': 0.0,
            # 'leg_l4_joint': 0.36,
            # 'leg_l5_joint': -0.175, 
            # 'leg_l6_joint': 0.0,

            # 'leg_r1_joint': -0.185,
            # 'leg_r2_joint': 0.0,
            # 'leg_r3_joint': 0.0,
            # 'leg_r4_joint': 0.36,
            # 'leg_r5_joint': -0.175,
            # 'leg_r6_joint': 0.0,

            # 'waist_yaw_joint': 0.0,

            # 'upper_left_1_joint': 0.0,
            # 'upper_left_2_joint': 0.0,
            # 'upper_left_3_joint': 0.0,
            # 'upper_left_4_joint': 0.0,
            # 'upper_left_5_joint': 0.0,
            # 'upper_left_6_joint': 0.0,
            # 'upper_left_7_joint': 0.0,


            # 'upper_right_1_joint': 0.0,
            # 'upper_right_2_joint': 0.0,
            # 'upper_right_3_joint': 0.0,
            # 'upper_right_4_joint': 0.0,
            # 'upper_right_5_joint': 0.0,
            # 'upper_right_6_joint': 0.0,
            # 'upper_right_7_joint': 0.0,

            # JT准备姿势
            # 'upper_left_1_joint': 0.046,
            # 'upper_left_2_joint': 0.269,
            # 'upper_left_3_joint': 0.392,
            # 'upper_left_4_joint': -0.898,
            # 'upper_left_5_joint': 1.325,
            # 'upper_left_6_joint': 0.23,
            # 'upper_left_7_joint': -1.57,


            # 'upper_right_1_joint': -0.522,
            # 'upper_right_2_joint': -0.97,
            # 'upper_right_3_joint': 0.833,
            # 'upper_right_4_joint': -0.849,
            # 'upper_right_5_joint': -1.175,
            # 'upper_right_6_joint': 0.504,
            # 'upper_right_7_joint': 1.148,

            # 'vhead_1_joint': 0.0,
            # 'vhead_2_joint': 0.0,

            #  0519：  music_BPM
            'leg_l1_joint':  -0.457,
            'leg_l2_joint': 0.192,
            'leg_l3_joint': 0.062,
            'leg_l4_joint': 0.874,
            'leg_l5_joint': -0.370, 
            'leg_l6_joint':  -0.126,

            'leg_r1_joint':  -0.457,
            'leg_r2_joint': -0.192,
            'leg_r3_joint': -0.062,
            'leg_r4_joint': 0.874,
            'leg_r5_joint': -0.370,
            'leg_r6_joint': 0.126,

            'waist_yaw_joint': 0.0,

            'upper_left_1_joint': 0.171,
            'upper_left_2_joint': 0.327,
            'upper_left_3_joint': 0.442,
            'upper_left_4_joint': -1.093,
            'upper_left_5_joint': 1.257,
            'upper_left_6_joint': -0.058,
            'upper_left_7_joint': -1.514,


            'upper_right_1_joint': -0.149,
            'upper_right_2_joint':  -0.755,
            'upper_right_3_joint': 0.306,
            'upper_right_4_joint': -1.865,
            'upper_right_5_joint': 0.259,
            'upper_right_6_joint': 0.156,
            'upper_right_7_joint': 1.073,

            'vhead_1_joint': 0.0,
            'vhead_2_joint': 0.0,
            

        }
    class control(LeggedRobotCfg.control):
        
        # PD Drive parameters: 腿/腰用电机常数，上肢/头固定值
        # stiffness = {
            # 'leg_l1_joint': STIFFNESS_10020_12,
            # 'leg_l2_joint': STIFFNESS_10020_12,
            # # 'leg_l2_joint': STIFFNESS_10020_24,
            # 'leg_l3_joint': STIFFNESS_6408_25,
            # # 'leg_l3_joint': STIFFNESS_8116,
            # 'leg_l4_joint': STIFFNESS_10020_12,
            # 'leg_l5_joint': STIFFNESS_6408_25 / 2,
            # 'leg_l6_joint': STIFFNESS_6408_25 / 2,

            # 'leg_r1_joint': STIFFNESS_10020_12,
            # 'leg_r2_joint': STIFFNESS_10020_12,
            # # 'leg_r2_joint': STIFFNESS_10020_24,
            # 'leg_r3_joint': STIFFNESS_6408_25,
            # # 'leg_r3_joint': STIFFNESS_8116,
            # 'leg_r4_joint': STIFFNESS_10020_12,
            # 'leg_r5_joint': STIFFNESS_6408_25 / 2,
            # 'leg_r6_joint': STIFFNESS_6408_25 / 2,

            # 'waist_yaw_joint': STIFFNESS_6408_25 ,
        stiffness = {
            'leg_l1_joint': STIFFNESS_10020_12 / 2,
            'leg_l2_joint': STIFFNESS_10020_12 / 2,
            # 'leg_l2_joint': STIFFNESS_10020_24,
            # 'leg_l3_joint': STIFFNESS_6408_25 / 2,
            # 'leg_l3_joint': STIFFNESS_8116,
            'leg_l3_joint': STIFFNESS_6416_25 / 2,
            'leg_l4_joint': STIFFNESS_10020_12 / 2,
            'leg_l5_joint': STIFFNESS_6408_25 / 2,
            'leg_l6_joint': STIFFNESS_6408_25 / 2,

            'leg_r1_joint': STIFFNESS_10020_12 / 2,
            'leg_r2_joint': STIFFNESS_10020_12 / 2,
            # 'leg_r2_joint': STIFFNESS_10020_24,
            # 'leg_r3_joint': STIFFNESS_6408_25 / 2,
            # 'leg_r3_joint': STIFFNESS_8116,
            'leg_r3_joint': STIFFNESS_6416_25 / 2,
            'leg_r4_joint': STIFFNESS_10020_12 / 2,
            'leg_r5_joint': STIFFNESS_6408_25 / 2,
            'leg_r6_joint': STIFFNESS_6408_25 / 2,

            # 'waist_yaw_joint': STIFFNESS_6408_25 / 2,
            'waist_yaw_joint': 200.0,

            'upper_left_1_joint': 200.0,
            'upper_left_2_joint': 200.0,
            'upper_left_3_joint': 200.0,
            'upper_left_4_joint': 200.0,
            'upper_left_5_joint': 200.0,
            'upper_left_6_joint': 200.0,
            'upper_left_7_joint': 200.0,


            'upper_right_1_joint': 200.0,
            'upper_right_2_joint': 200.0,
            'upper_right_3_joint': 200.0,
            'upper_right_4_joint': 200.0,
            'upper_right_5_joint': 200.0,
            'upper_right_6_joint': 200.0,
            'upper_right_7_joint': 200.0,

            'vhead_1_joint': 200.0,
            'vhead_2_joint': 200.0,
            
          }

        damping = {
            'leg_l1_joint': DAMPING_10020_12 / 2,
            'leg_l2_joint': DAMPING_10020_12 / 2,
            # 'leg_l2_joint': DAMPING_10020_24 / 2,
            # 'leg_l3_joint': DAMPING_6408_25 / 2,
            # 'leg_l3_joint': DAMPING_8116 / 2,
            'leg_l3_joint': DAMPING_6416_25 / 2,
            'leg_l4_joint': DAMPING_10020_12 /2,
            'leg_l5_joint': DAMPING_6408_25 / 2,    
            'leg_l6_joint': DAMPING_6408_25 / 2,

            'leg_r1_joint': DAMPING_10020_12 / 2,
            'leg_r2_joint': DAMPING_10020_12 / 2,
            # 'leg_r2_joint': DAMPING_10020_24 / 2,
            # 'leg_r3_joint': DAMPING_6408_25 / 2,
            # 'leg_r3_joint': DAMPING_8116 / 2,
            'leg_r3_joint': DAMPING_6416_25 / 2,
            'leg_r4_joint': DAMPING_10020_12 / 2,
            'leg_r5_joint': DAMPING_6408_25 / 2,
            'leg_r6_joint': DAMPING_6408_25 / 2,

            # 'waist_yaw_joint': DAMPING_6408_25 / 2,
            'waist_yaw_joint': 5.0,

            'upper_left_1_joint': 5.0,
            'upper_left_2_joint': 5.0,
            'upper_left_3_joint': 5.0,
            'upper_left_4_joint': 5.0,
            'upper_left_5_joint': 5.0,
            'upper_left_6_joint': 5.0,
            'upper_left_7_joint': 5.0,


            'upper_right_1_joint': 5.0,
            'upper_right_2_joint': 5.0,
            'upper_right_3_joint': 5.0,
            'upper_right_4_joint': 5.0,
            'upper_right_5_joint': 5.0,
            'upper_right_6_joint': 5.0,
            'upper_right_7_joint': 5.0,

            'vhead_1_joint': 5.0,
            'vhead_2_joint': 5.0,
            
        }
 
    
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # False: default_dof_pos + residual；True: ref_dof + residual（受控关节）
        use_ref_residual_target = False  # False
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        # 课程式域随机化：优先按训练稳定性指标自适应推进；必要时可退回 iteration 模式
        use_curriculum = True
        curriculum_mode = "adaptive"  # "adaptive" or "iteration"
        curriculum_stage_iters = [0, 700, 1400, 2100]
        push_ratio_schedule = [0.15, 0.35, 0.65, 1.0]
        disturbance_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        restitution_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        pd_ratio_schedule = [0.20, 0.40, 0.70, 1.0]
        ankle_pd_ratio_schedule = [0.20, 0.40, 0.70, 1.0]
        motor_strength_ratio_schedule = [0.20, 0.40, 0.70, 1.0]
        delay_ratio_schedule = [0.0, 0.25, 0.50, 1.0]
        imu_bias_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        motor_offset_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        ankle_motor_offset_ratio_schedule = [0.10, 0.25, 0.50, 1.0]


        payload_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        com_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        link_mass_ratio_schedule = [0.10, 0.25, 0.50, 1.0]
        
        
        
        adaptive_min_iteration = 500
        adaptive_metric_ema = 0.90
        adaptive_min_resets = 256  # 256
        adaptive_stage_cooldown_iterations = 100
        adaptive_length_ratio_thresholds = [0.0, 0.25, 0.45, 0.70]
        adaptive_fall_ratio_thresholds = [1.0, 0.55, 0.35, 0.20]

        randomize_payload_mass = True
        payload_mass_range = [-4.0,4]
        payload_body_name = "waist_yaw_link"  

        randomize_com_displacement = True
        # com_displacement_range = [-0.05, 0.05]
        com_body_name = "waist_yaw_link"  
        # 固定质心偏移
        com_offset_x = 0.0
        com_offset_y = 0.0
        com_offset_z = 0.0
        com_x_pos_range = [-0.08, 0.08]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.08, 0.08]

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]


        randomize_friction = True
        friction_range = [0.3, 1.4]  # [0.4, 2.0]

        randomize_restitution = True
        restitution_range = [0., 0.05]

        # 接触求解参数随机化：
        # 用来覆盖 MuJoCo/实物脚底接触刚度、穿透深度、碰撞 margin 与 IsaacGym 默认接触模型不一致的问题。
        randomize_contact_offsets = False
        contact_offset_range = [0.005, 0.02]
        rest_offset_range = [0.0, 0.002]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_motor_offset = False  # True
        motor_offset_range = [-0.015, 0.025]  # [-0.035, 0.045]

        # 默认关节角偏置随机化：
        # 用来模拟 default_dof_pos 本身存在小偏差，而不是在最终 PD target 上额外加 motor offset。
        # 普通关节：每个 reset 随机 U(-0.01, 0.01) rad。
        # 脚踝关节：每个 reset 随机 U(-0.1, 0.1) rad。
        randomize_default_dof_pos_offset = True
        default_dof_pos_offset_range = [-0.01, 0.01]
        default_dof_pos_offset_ankle_indices = [4, 5, 10, 11]
        default_dof_pos_offset_ankle_range = [-0.05, 0.05]

        # 脚踝观测偏置随机化：
        # 仅污染 actor 输入中的脚踝 q/dq，用来模拟串并联近似解算带来的测量零偏。
        # 不修改真实物理状态、奖励和参考轨迹。
        ankle_obs_joint_indices = [4, 5, 10, 11]
        randomize_ankle_obs_pos_bias = False
        ankle_obs_pos_bias_range = [-0.05,0.12]   # rad
        randomize_ankle_obs_vel_bias = False
        ankle_obs_vel_bias_range = [-0.3, 0.3]     # rad/s
        randomize_ankle_obs_vel_noise = False
        ankle_obs_vel_noise_std = 0.25             # rad/s, 只污染 actor 输入的脚踝 dq
        randomize_ankle_obs_vel_delay = False
        ankle_obs_vel_delay_range = [0, 5]         # control steps
        randomize_ankle_obs_vel_filter = False
        ankle_obs_vel_filter_cutoff_range = [6.0, 20.0]  # Hz

        # 脚踝 PD 速度反馈随机化：
        # 只影响 PD 阻尼项里的脚踝 dq，模拟实物/串并联解算后的速度反馈噪声、延迟和带宽限制。
        randomize_ankle_pd_dq_noise = False
        ankle_pd_dq_noise_std = 0.2                # rad/s
        randomize_ankle_pd_dq_delay = False
        ankle_pd_dq_delay_range = [0, 5]           # control steps
        randomize_ankle_pd_dq_filter = False
        ankle_pd_dq_filter_cutoff_range = [6.0, 20.0]  # Hz

        randomize_joint_friction = False # True
        joint_friction_range = [[0.03, 0.08],
                                [0.03, 0.08],
                                [0.03, 0.08],
                                [0.03, 0.08],
                                [0.0, 0.0],
                                [0.0, 0.0],

                                [0.03, 0.08],
                                [0.03, 0.08],
                                [0.03, 0.08],
                                [0.03, 0.08],
                                [0.0, 0.0],
                                [0.0, 0.0],

                                [0.03, 0.08],

                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],

                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                [0.0, 0.03],
                                
                                [0.0, 0.03],
                                [0.0, 0.03],]

        randomize_joint_armature = False
        joint_armature_values = [ARMATURE_10020_12,  # leg_l1
                                ARMATURE_10020_12,  # leg_l2
                                # ARMATURE_6408_25,   # leg_l3
                                ARMATURE_6416_25,   # leg_l3_6416
                                ARMATURE_10020_12,  # leg_l4
                                ARMATURE_6408_25,   # leg_l5
                                ARMATURE_6408_25,   # leg_l6

                                ARMATURE_10020_12,  # leg_r1
                                ARMATURE_10020_12,  # leg_r2
                                # ARMATURE_6408_25,   # leg_r3
                                ARMATURE_6416_25,   # leg_r3_6416
                                ARMATURE_10020_12,  # leg_r4
                                ARMATURE_6408_25,   # leg_r5
                                ARMATURE_6408_25,   # leg_r6

                                ARMATURE_6408_25,   # waist_yaw

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
                                ARMATURE_UPPER]
        joint_armature_range = [ARMATURE_10020_12_RANGE,  # leg_l1
                                ARMATURE_10020_12_RANGE,  # leg_l2
                                # ARMATURE_10020_24_RANGE,  # leg_l2
                                # ARMATURE_6408_25_RANGE,   # leg_l3
                                # ARMATURE_8116_RANGE,   # leg_l3
                                ARMATURE_6416_25_RANGE,   # leg_l3
                                ARMATURE_10020_12_RANGE,  # leg_l4
                                ARMATURE_6408_25_RANGE,   # leg_l5
                                ARMATURE_6408_25_RANGE,   # leg_l6

                                ARMATURE_10020_12_RANGE,  # leg_r1
                                ARMATURE_10020_12_RANGE,  # leg_r2
                                # ARMATURE_10020_24_RANGE,  # leg_r2
                                # ARMATURE_6408_25_RANGE,   # leg_r3
                                # ARMATURE_8116_RANGE,   # leg_r3
                                ARMATURE_6416_25_RANGE,   # leg_r3
                                ARMATURE_10020_12_RANGE,  # leg_r4
                                ARMATURE_6408_25_RANGE,   # leg_r5
                                ARMATURE_6408_25_RANGE,   # leg_r6

                                ARMATURE_6408_25_RANGE,   # waist_yaw

                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,

                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,

                                ARMATURE_UPPER_RANGE,
                                ARMATURE_UPPER_RANGE,]

        disturbance = True
        # disturbance_range = [-500.0, 500.0]
        disturbance_range = [-100.0, 100.0]
        disturbance_s = 8

        push_robots = True
        push_interval_s = [1.0, 3.0]  # 每次 push 后在 1~3s 内随机下一次间隔
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.15

        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]

        randomize_ankle_pd = False
        ankle_joint_indices = [4, 5, 10, 11]
        ankle_kp_range = [0.75, 1.25]
        ankle_kd_range = [0.75, 1.25]

        randomize_ankle_motor_offset = False
        ankle_motor_offset_range = [-0.02, 0.02]

        action_delay = True
        action_delay_range = [5, 20]  # [5, 20]

        sys_delay = False   # eps reset
        imu_delay_range = [1, 5]  #  [1, 5]
        motor_delay_range = [5, 20]  #  [5, 20]

        use_coulomb = False
        left_Us = 2.8
        left_Qs = 0.1
        left_Ud = 0.21
        right_Us = 2.8
        right_Qs = 0.1
        right_Ud = 0.21
        star_Us = 0.45
        star_Qs = 0.1
        star_Ud = 0.023

        randomize_upperbody_speed = False
        upperbody_speed_range = [0, 12]

        randomize_euler_xy_offset = False  # False
        euler_xy_offset_range = [-0.02, 0.02]

        randomize_euler_z_offset = False
        euler_z_offset_range = [-3.14, 3.14]

        # “初始状态噪声”：
        # 1. xy reset 噪声：在参考帧的平面位置附近随机平移一点，
        #    防止策略只会在某个固定落点上工作，提高对落脚点/站位误差的鲁棒性。
        # 2. yaw reset 噪声：在参考朝向附近随机转一点，
        #    防止策略只会在某个精确朝向上起跳/起步，提高对初始偏航误差的适应能力。
        randomize_root_xy_reset = True
        root_xy_reset_range = [-0.05, 0.05] 
        randomize_root_yaw_reset = True
        root_yaw_reset_range = [-0.17, 0.17] 

        # 小幅初始关节姿态随机化（接近 qpos0 / init qpos 的思路）：
        # 在 RSI 对齐到参考关节姿态之后，再给各关节叠加一层较小的随机扰动
        randomize_init_dof_pos = True
        init_dof_pos_range = [-0.03, 0.03]  

        RSI = 1

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 1.04
        min_dist = 0.25
        max_dist = 2.0
         
        # hip_pos_scale = 0.3    # rad
        # knee_pos_scale = 0.3
        # ankle_pos_scale = 0.2
        
        # put some settings here for LLM parameter tuning
        # target_joint_pos_scale = 0.17    # rad
        # target_feet_height = 0.06       # m
        # cycle_time = 11.37                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 600  # Forces above this value are penalized
        feet_air_time_target = 0.4  # 目标腾空时间 [s]，接近时给奖励，仅在有水平速度时计奖
        # 12 个受控腿部关节的 imitation joint pos/vel 误差权重。
        # 降低脚踝 pitch/roll 权重，避免策略为了死跟踪脚踝角度而用踝关节高频补偿接触扰动。
        dof_err_w = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]

        class sigma:
            foot_height         = 0.08  # exp(-diff^2 / sigma^2)，越小越严格
            com_over_support_foot = 0.08  # 质心-支撑脚距离 exp(-d^2/sigma^2)，越小越严格
            feet_pos            = 0.15  # exp(-dist_sq / sigma^2)
            feet_rot            = 0.2  # exp(-rot_err^2 / sigma^2)
            whole_body_pos      = 0.15    # exp(-dist_sq / sigma^2)  0.3  0.1
            whole_body_rot      = 0.2   # exp(-rot_err^2 / sigma^2)  0.4  0.15
            whole_body_lin_vel  = 1.0    # exp(-vel_err / sigma^2)  1
            whole_body_ang_vel  = 3.14   # exp(-ang_vel_err / sigma^2)  3.14  10
            root_height         = 0.3   # exp(-diff_sq / sigma^2)  2cm误差时约0.95分
            root_pos            = 0.3    # exp(-diff_sq / sigma^2)  0.3
            root_rot            = 0.4    # exp(-rot_err^2 / sigma^2)  0.4
            root_ang_vel        = 1.5   # exp(-diff_sq / sigma^2)  机身角速度跟踪
            root_vel            = 0.6    # exp(-diff_sq / sigma^2)  0.4  0.1

        class scales:
            # ------------ mimic ------------
            # tracking_lin_vel = 2.6
            # tracking_ang_vel = 2.1
            # imition_root_pos_xyz = 5
            # imition_torso_orientation = 0.5
            # imition_torso_yaw = 0.5  # 5
            # imition_linear_velocity_x = 0.5  #root 
            # imition_linear_velocity_y = 0.5
            # imition_linear_velocity_z = 0.5
            # imition_angular_velocity_xy = 0.5
            # imition_angular_velocity_z = 0.5  # 1.0
            # imition_joint_pos_leg = 3.
            # # imition_leg_joint_pos_error_exp = 0
            
            # imition_leg_joint_vel_error_exp = 0
            # imition_keybody_pos = 10.
            # imition_keybody_euler = 5
            # imition_keybody_vel = 2
            # imition_keybody_orientation = 5
            # imition_knee_rot = 0 # 0.5
            # imition_keybody_ang_vel = 2  # 2
            # imition_keybody_lin_vel = 2  #2
            # imition_head_pos = 0.5 # 2
            # imition_head_rot = 0  # 2
            # imition_waist_pos = 2 # 2
            # imition_waist_rot = 1  # 2
            
            
            # imition_base_lin_vel = 1
            # imition_base_ang_vel = 0.5
            # imition_root_pos = 0.3
            
            # imition_knee_pos = 2 # 1
            # imition_hip_pos = 1.5 # 1
            # imition_hip_rot = 1.5 #  0.5
            # root_xy_stay = 1.5  # 惩罚机身 XY 漂移，使机器人尽量原地舞蹈
            # root_rot_stay = 1.5  # 惩罚机身旋转偏离参考，保持朝向一致
            # torso_yaw_stay = 1  # 惩罚机身锚点偏航角偏离参考
            # imition_base_ang_vel = 5.5  # 跟随机身 root 角速度
            # imition_joint_pos = 0.5
            # feet_height = 5
            # imition_joint_vel = 0.2
            
            # forward_lean = 2.5
            # imition_foot_height = 1
            # imition_feet_pos = 0.2
            # imition_feet_rot = 0.5
            # imition_root_height = 0.3
            imitation_whole_body_ang_vel = 1
            imitation_whole_body_lin_vel = 1
            imitation_whole_body_rot = 1
            imitation_whole_body_pos = 1
            imition_root_pos = 0.5
            imition_root_rot = 0.5
            # imitation_root_vel = 0.2
            # imition_base_ang_vel = 0.3
            # teleop_contact_mask = 1
            # ref_lift_foot_clearance = 6.5
            # swing_feet_height = 1
            # com_over_support_foot = 0.5
            termination = -50.0     
            # ------------- regularization ----------------------          
            # foot_slip = -1
            # stance_foot_slip = -1.0  # 惩罚支撑脚滑动：不滑=0，有滑=惩罚
            # stance_feet_speed_reg = -0.15  # 惩罚支撑脚 yaw 角速度过大，压制脚底在地上打转
            # landing_foot_z_vel = -0.05  # 惩罚摆动脚第一次落地瞬间 z 方向速度过大，鼓励柔和落脚
            # pre_landing_foot_z_vel = -0.08  # 摆动脚近地未接触时，惩罚向下速度过大，避免砸地
            # pre_landing_foot_smooth = -0.05  # 落地前近地阶段惩罚足底 roll/pitch 角速度和姿态误差
            # dof_acc = -5e-6  #  -5e
            # dof_vel = -1e-3
            action_rate = -0.01
            # waist_action = -0.1  # 惩罚腰关节 residual 动作幅度，减少用腰补偿
            # ankle_dof_acc = -1e-6  # -5e-8
            ankle_dof_vel = -1e-3  # -5e-5
            # ankle_roll_dof_vel = -1e-3   # 只惩罚左右 ankle roll 速度，优先压制内外侧支撑抖动
            # action_smoothness = -0.02
            dof_pos_limits = -3.
            torque_limits = -2
            ankle_torque_limit = -3 # 惩罚踝关节力矩超限
            # feet_contact_forces = -0.02 # -0.01  -5e-3
            # torques = -1e-6  # -1e-5
            # penalty_stumble = -1 
            # contact_no_vel = -60.0  # -60
            # feet_contact_orientation = -0.3  # -100
            # torques_smoothness = -1e-4
            # ankle_regularization = -1  
            # ankle_action_rate = -0.02
            # lin_vel_z = -2.0
            
            
            # ang_vel_xy = -0.05  # -0.05
            # orientation = -0.3  # -0.1
            # collision = -0.1
            # energy = -5e-5   # -5e-5
            # dof_vel_limits = -1
            
            # torque_penalty = -1

            # --------- gait -------------
            # feet_clearance = 1.
            # feet_contact_number = 1.2
            # feet_air_time = 5.
            # feet_distance = 0.5
            # knee_distance = 0.2
            # vel_mismatch_exp = 0.3  # lin_z; ang x,y
            # stand = -0.2
            # low_speed = 0.5
            # track_vel_hard = 0.5
            # default_joint_pos = 0.2
            # orientation = 1.
            # base_acc = 0.2
            # foot_pitch = 1.0
            # imition_mirr_joint_pos = 2.
            # imition_joint_pos_arm = 5.
            # imition_contact = 1.
            # imition_survival = 1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
            com_pos = 10

        clip_observations = 50
        clip_actions = 50
        actions_filter = True

class MrobotMimicCommonCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        # fixed_std = True
        init_noise_std = np.array([0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 
                                   0.8, 0.8, 0.8, 0.8, 1.2, 1.2])*1
        # init_noise_std = -1.
        num_single_obs = 45
        num_goal_obs = 19  # 与 env.num_goal_obs 保持一致
        actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [768, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # entropy_coef = 0.001
        # learning_rate = 5e-5
        # num_learning_epochs = 5
        # gamma = 0.995
        # lam = 0.95
        # num_mini_batches = 4
        normalizer_update_iterations = 2400
        entropy_coef = 0.005  # 0.005
        learning_rate = 1e-4  # 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iterationxunl1
        max_iterations = 53000  # number of policy updates

        # logging
        save_interval =250  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'mrobot_mimic_May_music_BPM'
        run_name = ''
        # Load and resume
        resume = False 
        load_run = 'May28_17-21-35_'   # JT:'Jan30_10-14-21_'  # -1 = last run  Feb05_18-53-17_
        checkpoint = -1  # -1 = last saved model
        resume_path = None
        # resume_path = 'logs/mrobot_mimic_X1/Feb26_19-33-55_/model_6250.pt'  # updated from load_run and chkpt
        save_config = 'mrobot_mimic_common_config.py'
        # JT_2min37s_1000HZ_keypoint
