"""Shared IsaacLab config base for MRobot mimic tasks.

This module contains Lab/IsaacSim-specific pieces that are common to both
parallel tasks:

- ``mrobot_music`` / BPM reference-network mimic.
- ``mrobot_dance`` / specified-trajectory mimic.

Task-specific Lab configs should inherit this base and then override
``env/motion/rewards`` from their own task config.
"""

from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_config import (
    MrobotMimicCommonCfg,
    MrobotMimicCommonCfgPPO,
)


class MrobotMimicCommonLabCfg(MrobotMimicCommonCfg):
    lab_joint_effort_limits = {
        "leg_l1_joint": 74.4,
        "leg_l2_joint": 74.4,
        "leg_l3_joint": 64.4,
        "leg_l4_joint": 74.4,
        "leg_l5_joint": 41.5,
        "leg_l6_joint": 41.5,
        "leg_r1_joint": 74.4,
        "leg_r2_joint": 74.4,
        "leg_r3_joint": 64.4,
        "leg_r4_joint": 74.4,
        "leg_r5_joint": 41.5,
        "leg_r6_joint": 41.5,
        "waist_yaw_joint": 41.5,
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


class MrobotMimicCommonLabCfgPPO(MrobotMimicCommonCfgPPO):
    pass


__all__ = [
    "MrobotMimicCommonLabCfg",
    "MrobotMimicCommonLabCfgPPO",
]
