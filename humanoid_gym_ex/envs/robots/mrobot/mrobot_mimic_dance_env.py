from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch

from humanoid_gym_ex.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid_gym_ex.envs.robots.mrobot.mrobot_legged_robot import LeggedRobot, get_euler_xyz_tensor
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_env import MrobotMimicEnv
from humanoid_gym_ex.utils.mrobot_trajectory_reference import get_motion_files_from_cfg, load_mrobot_trajectory_library


class MrobotMimicDanceEnv(MrobotMimicEnv):
    """IsaacGym MRobot mimic task driven by specified trajectory ``.npz`` files."""

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        LeggedRobot.__init__(self, cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.num_aux = self.cfg.env.num_aux
        self.rand_init_coef = self.cfg.env.rand_init_coef
        self.num_disc_obs = self.cfg.env.num_disc_obs
        dof_err_w = getattr(self.cfg.rewards, "dof_err_w", [1.0] * len(self.num_control))
        if len(dof_err_w) != len(self.num_control):
            raise ValueError(
                f"cfg.rewards.dof_err_w length must be {len(self.num_control)}, got {len(dof_err_w)}"
            )
        self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)

        self._load_trajectory_library()
        self._init_reference_state_buffers()
        self.phase_rad = torch.zeros(self.num_envs, 1, device=self.device)
        self.normalized_bpm_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        print("[mrobot_dance] 指定轨迹 reference 加载完成")
        print(f"[mrobot_dance] 轨迹数量: {self.data_length}")
        print(f"[mrobot_dance] 各轨迹真实长度: {[int(x) for x in self.demo_lengths.detach().cpu().tolist()]}")

        self.last_root_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.all_tracking_indices = torch.cat(
            (
                self.base_indices,
                self.feet_indices,
                self.knee_indices,
                self.hip_indices,
                self.pelvic_yaw_indices,
                self.waist_indices,
            ),
            dim=0,
        ).long()
        self.is_static_stand = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.base_height_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ref_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_root_offset = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float)
        self.initial_base_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self._ankle_obs_joint_indices = torch.tensor(
            getattr(self.cfg.domain_rand, "ankle_obs_joint_indices", [4, 5, 10, 11]),
            device=self.device,
            dtype=torch.long,
        )
        n_ankle_obs = len(self._ankle_obs_joint_indices)
        self.ankle_obs_pos_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_obs_vel_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self._init_ankle_dq_randomization_buffers(n_ankle_obs)
        self._build_hard_phase_windows()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    def _load_trajectory_library(self):
        motion_cfg = getattr(self.cfg, "motion", None)
        motion_files = get_motion_files_from_cfg(motion_cfg)
        library = load_mrobot_trajectory_library(
            motion_files,
            self.device,
            allow_legacy_keypoint_fallback=bool(getattr(motion_cfg, "allow_legacy_keypoint_fallback", False)),
        )
        self.motion_files = library.files
        self.data_length = library.data_length
        self.demo_length = library.demo_length
        self.demo_lengths = library.demo_lengths
        buffers = library.buffers
        self.dof_pos_buffer = buffers["dof_pos"]
        self.dof_vel_buffer = buffers["dof_vel"]
        self.root_states_buffer = buffers["root_states"]
        self.root_linvel_buffer = buffers["root_linvel"]
        self.root_angvel_buffer = buffers["root_angvel"]
        self.euler_xyz_buffer = buffers["euler_xyz"]
        self.foot_height_buffer = buffers["foot_height"]
        self.feet_contact_buffer = buffers["feet_contact"]
        self.pelvis_pos_buffer = buffers["pelvis_pos"]
        self.pelvis_vel_buffer = buffers["pelvis_vel"]
        self.pelvis_quat_buffer = buffers["pelvis_quat"]
        self.pelvis_ang_vel_buffer = buffers["pelvis_ang_vel"]
        self.feet_pos_buffer = buffers["feet_pos"]
        self.feet_vel_buffer = buffers["feet_vel"]
        self.feet_quat_buffer = buffers["feet_quat"]
        self.feet_ang_vel_buffer = buffers["feet_ang_vel"]
        self.knee_pos_buffer = buffers["knee_pos"]
        self.knee_vel_buffer = buffers["knee_vel"]
        self.knee_quat_buffer = buffers["knee_quat"]
        self.knee_ang_vel_buffer = buffers["knee_ang_vel"]
        self.hip_pos_buffer = buffers["hip_pos"]
        self.hip_vel_buffer = buffers["hip_vel"]
        self.hip_quat_buffer = buffers["hip_quat"]
        self.hip_ang_vel_buffer = buffers["hip_ang_vel"]
        self.pelvic_yaw_pos_buffer = buffers["pelvic_yaw_pos"]
        self.pelvic_yaw_vel_buffer = buffers["pelvic_yaw_vel"]
        self.pelvic_yaw_quat_buffer = buffers["pelvic_yaw_quat"]
        self.pelvic_yaw_ang_vel_buffer = buffers["pelvic_yaw_ang_vel"]
        self.waist_pos_buffer = buffers["waist_pos"]
        self.waist_vel_buffer = buffers["waist_vel"]
        self.waist_quat_buffer = buffers["waist_quat"]
        self.waist_ang_vel_buffer = buffers["waist_ang_vel"]

    def compute_ref_state(self):
        max_phase = self.demo_lengths[self.ref_idx].clamp(min=1) - 1
        phase_ids = torch.minimum(self.phase_idx, max_phase)
        self.ref_dof_pos[:] = self.dof_pos_buffer[self.ref_idx, phase_ids]
        self.ref_dof_vel[:] = self.dof_vel_buffer[self.ref_idx, phase_ids]
        self.ref_pelvis_pos[:] = self.pelvis_pos_buffer[self.ref_idx, phase_ids]
        self.ref_pelvis_vel[:] = self.pelvis_vel_buffer[self.ref_idx, phase_ids]
        self.ref_pelvis_quat[:] = self.pelvis_quat_buffer[self.ref_idx, phase_ids]
        self.ref_pelvis_ang_vel[:] = self.pelvis_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_feet_pos[:] = self.feet_pos_buffer[self.ref_idx, phase_ids]
        self.ref_feet_vel[:] = self.feet_vel_buffer[self.ref_idx, phase_ids]
        self.ref_feet_quat[:] = self.feet_quat_buffer[self.ref_idx, phase_ids]
        self.ref_feet_ang_vel[:] = self.feet_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_knee_pos[:] = self.knee_pos_buffer[self.ref_idx, phase_ids]
        self.ref_knee_vel[:] = self.knee_vel_buffer[self.ref_idx, phase_ids]
        self.ref_knee_quat[:] = self.knee_quat_buffer[self.ref_idx, phase_ids]
        self.ref_knee_ang_vel[:] = self.knee_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_hip_pos[:] = self.hip_pos_buffer[self.ref_idx, phase_ids]
        self.ref_hip_vel[:] = self.hip_vel_buffer[self.ref_idx, phase_ids]
        self.ref_hip_quat[:] = self.hip_quat_buffer[self.ref_idx, phase_ids]
        self.ref_hip_ang_vel[:] = self.hip_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_pelvic_yaw_pos[:] = self.pelvic_yaw_pos_buffer[self.ref_idx, phase_ids]
        self.ref_pelvic_yaw_vel[:] = self.pelvic_yaw_vel_buffer[self.ref_idx, phase_ids]
        self.ref_pelvic_yaw_quat[:] = self.pelvic_yaw_quat_buffer[self.ref_idx, phase_ids]
        self.ref_pelvic_yaw_ang_vel[:] = self.pelvic_yaw_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_waist_pos[:] = self.waist_pos_buffer[self.ref_idx, phase_ids]
        self.ref_waist_vel[:] = self.waist_vel_buffer[self.ref_idx, phase_ids]
        self.ref_waist_quat[:] = self.waist_quat_buffer[self.ref_idx, phase_ids]
        self.ref_waist_ang_vel[:] = self.waist_ang_vel_buffer[self.ref_idx, phase_ids]
        self.ref_feet_contact[:] = self.feet_contact_buffer[self.ref_idx, phase_ids]
        self.ref_foot_height[:] = self.foot_height_buffer[self.ref_idx, phase_ids]
        self.ref_root_linvel[:] = self.root_linvel_buffer[self.ref_idx, phase_ids]
        self.ref_root_angvel[:] = self.root_angvel_buffer[self.ref_idx, phase_ids]
        self.ref_euler_xyz[:] = self.euler_xyz_buffer[self.ref_idx, phase_ids]

    def _get_actor_reference_extra_obs(self):
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _get_privileged_reference_phase_obs(self, norm_phase):
        return torch.zeros_like(norm_phase)

    def _advance_reference_phase(self):
        LeggedRobot._advance_reference_phase(self)

    def post_physics_step(self):
        LeggedRobot.post_physics_step(self)

    def check_termination(self):
        LeggedRobot.check_termination(self)

    def _reset_dofs(self, env_ids):
        LeggedRobot._reset_dofs(self, env_ids)

    def _reset_root_states(self, env_ids):
        ref_ids = self.ref_idx[env_ids]
        phase_ids = self.episode_phase_buf[env_ids]
        ref_root = self.root_states_buffer[ref_ids, phase_ids]
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] = self.env_origins[env_ids] + ref_root[:, :3]
        self.root_states[env_ids, 3:7] = ref_root[:, 3:7]
        self.root_states[env_ids, 7:10] = self.root_linvel_buffer[ref_ids, phase_ids]
        self.root_states[env_ids, 10:13] = self.root_angvel_buffer[ref_ids, phase_ids]

        if getattr(self.cfg.domain_rand, "randomize_root_xy_reset", False):
            xy_range = getattr(self.cfg.domain_rand, "root_xy_reset_range", [0.0, 0.0])
            self.root_states[env_ids, 0:2] += torch_rand_float(
                xy_range[0], xy_range[1], (len(env_ids), 2), device=self.device
            )
        if getattr(self.cfg.domain_rand, "randomize_root_yaw_reset", False):
            yaw_range = getattr(self.cfg.domain_rand, "root_yaw_reset_range", [0.0, 0.0])
            yaw_noise = torch_rand_float(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=self.device)[:, 0]
            zero_tensor = torch.zeros_like(yaw_noise)
            yaw_noise_quat = quat_from_euler_xyz(zero_tensor, zero_tensor, yaw_noise)
            self.root_states[env_ids, 3:7] = quat_mul(yaw_noise_quat, self.root_states[env_ids, 3:7])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        LeggedRobot.reset_idx(self, env_ids)
        if len(env_ids) == 0:
            return
        self.phase_idx[env_ids] = self.episode_phase_buf[env_ids]
        self.compute_ref_state()
        self._resample_ankle_obs_bias(env_ids)
        self._resample_ankle_dq_randomization(env_ids)
        self.initial_base_yaw[env_ids] = get_euler_xyz_tensor(self.root_states[env_ids, 3:7])[:, 2]
