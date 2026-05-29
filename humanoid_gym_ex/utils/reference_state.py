import torch
import torch.nn as nn


JOINT_NAME_ALIASES = {
    "left_leg_pelvic_pitch": "leg_l1_joint",
    "left_leg_pelvic_roll": "leg_l2_joint",
    "left_leg_pelvic_yaw": "leg_l3_joint",
    "left_leg_knee_pitch": "leg_l4_joint",
    "left_leg_ankle_pitch": "leg_l5_joint",
    "left_leg_ankle_roll": "leg_l6_joint",
    "right_leg_pelvic_pitch": "leg_r1_joint",
    "right_leg_pelvic_roll": "leg_r2_joint",
    "right_leg_pelvic_yaw": "leg_r3_joint",
    "right_leg_knee_pitch": "leg_r4_joint",
    "right_leg_ankle_pitch": "leg_r5_joint",
    "right_leg_ankle_roll": "leg_r6_joint",
    "waist_yaw": "waist_yaw_joint",
    "left_shoulder_pitch": "upper_left_1_joint",
    "left_shoulder_roll": "upper_left_2_joint",
    "left_shoulder_yaw": "upper_left_3_joint",
    "left_elbow_pitch": "upper_left_4_joint",
    "left_wrist_yaw": "upper_left_5_joint",
    "left_wrist_pitch": "upper_left_6_joint",
    "left_wrist_roll": "upper_left_7_joint",
    "right_shoulder_pitch": "upper_right_1_joint",
    "right_shoulder_roll": "upper_right_2_joint",
    "right_shoulder_yaw": "upper_right_3_joint",
    "right_elbow_pitch": "upper_right_4_joint",
    "right_wrist_yaw": "upper_right_5_joint",
    "right_wrist_pitch": "upper_right_6_joint",
    "right_wrist_roll": "upper_right_7_joint",
}


class ReferenceStateNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super().__init__()
        layers = []
        last_dim = input_dim
        for width in hidden:
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.SiLU())
            last_dim = width
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def encode_bpm_phase(bpm_cmd, phase_rad, bpm_mean, bpm_std):
    bpm_mean = torch.as_tensor(bpm_mean, device=bpm_cmd.device, dtype=bpm_cmd.dtype)
    bpm_std = torch.as_tensor(bpm_std, device=bpm_cmd.device, dtype=bpm_cmd.dtype).clamp(min=1e-6)
    return torch.cat(
        (
            (bpm_cmd - bpm_mean) / bpm_std,
            torch.sin(phase_rad),
            torch.cos(phase_rad),
        ),
        dim=-1,
    )
