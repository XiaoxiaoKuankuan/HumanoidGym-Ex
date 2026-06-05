from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import torch

from humanoid_gym_ex.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid_gym_ex.envs.robots.mrobot.mrobot_legged_robot import LeggedRobot, get_euler_xyz_tensor
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_common_env import MrobotMimicCommonEnv
from humanoid_gym_ex.utils.mrobot_trajectory_reference import get_motion_files_from_cfg, load_mrobot_trajectory_library


class MrobotMimicDanceEnv(MrobotMimicCommonEnv):
    """IsaacGym MRobot mimic task driven by specified trajectory ``.npz`` files."""

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self._apply_reference_fps_decimation(cfg, sim_params)
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
        self.waist_body_id = int(self.waist_indices[0].item()) if len(self.waist_indices) else int(self.base_indices[0].item())
        self.contact_reset_buf = torch.zeros_like(self.fall_reset_buf)
        self.base_too_low_buf = torch.zeros_like(self.fall_reset_buf)
        self.tracking_error_reset_buf = torch.zeros_like(self.fall_reset_buf)
        self.waist_z_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.waist_ori_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.foot_z_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.adaptive_phase_failure_buf = torch.zeros_like(self.fall_reset_buf)

        self._ankle_obs_joint_indices = torch.tensor(
            getattr(self.cfg.domain_rand, "ankle_obs_joint_indices", [4, 5, 10, 11]),
            device=self.device,
            dtype=torch.long,
        )
        n_ankle_obs = len(self._ankle_obs_joint_indices)
        self.ankle_obs_pos_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self.ankle_obs_vel_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device, dtype=torch.float)
        self._init_ankle_dq_randomization_buffers(n_ankle_obs)
        self._init_adaptive_phase_sampling()

        reference_fps = float(getattr(self.cfg.motion, "reference_fps", 0.0))
        expected_ref_dt = 1.0 / reference_fps if reference_fps > 0.0 else 0.0
        print(
            "[mrobot_dance][IsaacGym] timing: "
            f"sim.dt={self.sim_params.dt:.6f}, control.decimation={self.cfg.control.decimation}, "
            f"policy_dt={self.dt:.6f}, reference_fps={reference_fps:.3f}, "
            f"expected_reference_dt={expected_ref_dt:.6f}, "
            f"physics_substeps_per_rollout={self.cfg.env.num_steps_per_env * self.cfg.control.decimation if hasattr(self.cfg.env, 'num_steps_per_env') else 'n/a'}",
            flush=True,
        )
        print(
            "[mrobot_dance][IsaacGym] adaptive phase sampling: "
            f"enabled={getattr(self.cfg.motion, 'use_adaptive_phase_sampling', False)}, "
            f"zero_start_ratio={getattr(self.cfg.motion, 'zero_start_ratio', 0.0)}, "
            f"bin_size_frames={self.motion_bin_size_frames}, "
            f"num_bins={[int(x) for x in self.motion_num_bins.detach().cpu().tolist()]}",
            flush=True,
        )

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    @staticmethod
    def _apply_reference_fps_decimation(cfg, sim_params):
        if not bool(getattr(cfg.control, "match_reference_fps", False)):
            cfg.control.decimation = max(1, int(getattr(cfg.control, "decimation", 1)))
            return
        reference_fps = float(getattr(cfg.motion, "reference_fps", 0.0))
        sim_dt = float(getattr(sim_params, "dt", getattr(cfg.sim, "dt", 0.0)))
        if reference_fps <= 0.0 or sim_dt <= 0.0:
            cfg.control.decimation = max(1, int(getattr(cfg.control, "decimation", 1)))
            return
        raw_decimation = 1.0 / (reference_fps * sim_dt)
        matched_decimation = max(1, int(round(raw_decimation)))
        if not math.isclose(raw_decimation, float(matched_decimation), rel_tol=0.0, abs_tol=1e-6):
            print(
                "[mrobot_dance][IsaacGym][WARN] reference_fps does not divide sim.dt exactly: "
                f"reference_fps={reference_fps}, sim_dt={sim_dt}, raw_decimation={raw_decimation:.6f}, "
                f"using decimation={matched_decimation}",
                flush=True,
            )
        cfg.control.decimation = matched_decimation

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
        max_phase = self.demo_lengths[self.ref_idx].clamp(min=1) - 1
        self.phase_idx[:] = torch.minimum(self.phase_idx + 1, max_phase)

    def _init_adaptive_phase_sampling(self):
        motion = self.cfg.motion
        reference_fps = float(getattr(motion, "reference_fps", 1.0))
        bin_size_sec = float(getattr(motion, "adaptive_bin_size_sec", 1.0))
        self.motion_bin_size_frames = max(1, int(round(reference_fps * bin_size_sec)))
        self.motion_num_bins = torch.ceil(self.demo_lengths.float() / float(self.motion_bin_size_frames)).long().clamp(min=1)
        max_bins = int(torch.max(self.motion_num_bins).item())
        bin_ids = torch.arange(max_bins, dtype=torch.long, device=self.device).unsqueeze(0)
        self.motion_valid_bin_mask = bin_ids < self.motion_num_bins.unsqueeze(1)
        self.motion_bin_failed_count = torch.zeros(self.data_length, max_bins, device=self.device)
        self._current_motion_bin_failed = torch.zeros_like(self.motion_bin_failed_count)
        self.motion_sampling_prob = torch.zeros_like(self.motion_bin_failed_count)
        self.motion_sampling_entropy = torch.ones((), device=self.device)
        self.motion_sampling_top1_prob = torch.zeros((), device=self.device)
        self.motion_sampling_top1_bin = torch.zeros((), device=self.device)
        self._refresh_motion_sampling_prob()

    def _refresh_motion_sampling_prob(self):
        motion = self.cfg.motion
        valid = self.motion_valid_bin_mask
        num_valid = self.motion_num_bins.float().clamp(min=1.0)
        uniform = valid.float() / num_valid.unsqueeze(1)
        counts = self.motion_bin_failed_count * valid.float()
        failure_sum = counts.sum(dim=1, keepdim=True)
        adaptive_prob = uniform
        if torch.any(failure_sum > 0.0):
            kernel_size = max(1, int(getattr(motion, "adaptive_kernel_size", 3)))
            adaptive_lambda = float(getattr(motion, "adaptive_lambda", 0.8))
            if kernel_size > 1:
                left = kernel_size // 2
                right = kernel_size - 1 - left
                offsets = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - float(left)
                kernel = torch.pow(torch.full_like(offsets, adaptive_lambda), torch.abs(offsets))
                kernel = kernel / kernel.sum().clamp(min=1e-6)
                padded = torch.nn.functional.pad(counts.unsqueeze(1), (left, right), mode="replicate")
                smooth = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).squeeze(1)
            else:
                smooth = counts
            smooth = smooth * valid.float()
            smooth_sum = smooth.sum(dim=1, keepdim=True)
            adaptive_prob = torch.where(smooth_sum > 0.0, smooth / smooth_sum.clamp(min=1e-6), uniform)
        uniform_ratio = float(getattr(motion, "adaptive_uniform_ratio", 0.1))
        prob = (1.0 - uniform_ratio) * adaptive_prob + uniform_ratio * uniform
        prob = prob * valid.float()
        prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
        no_fail = failure_sum <= 0.0
        prob = torch.where(no_fail, uniform, prob)
        self.motion_sampling_prob[:] = prob
        entropy = -(prob * (prob + 1e-12).log()).sum(dim=1) / torch.log(num_valid.clamp(min=2.0))
        self.motion_sampling_entropy = entropy.mean()
        pmax, imax = prob.max(dim=1)
        top_motion = torch.argmax(pmax)
        self.motion_sampling_top1_prob = pmax[top_motion]
        self.motion_sampling_top1_bin = imax[top_motion].float()

    def _update_adaptive_phase_failures(self, env_ids):
        motion = self.cfg.motion
        if not bool(getattr(motion, "use_adaptive_phase_sampling", False)):
            return
        if len(env_ids) == 0:
            return
        self._current_motion_bin_failed.zero_()
        failure_mask = self.adaptive_phase_failure_buf[env_ids]
        if not torch.any(failure_mask):
            return
        failed_envs = env_ids[failure_mask]
        ref_ids = self.ref_idx[failed_envs].clamp(min=0, max=max(self.data_length - 1, 0))
        phase_ids = self.phase_idx[failed_envs].clamp(min=0)
        bin_ids = torch.div(phase_ids, self.motion_bin_size_frames, rounding_mode="floor")
        bin_ids = torch.minimum(bin_ids, self.motion_num_bins[ref_ids] - 1).clamp(min=0)
        flat_count = self.motion_bin_failed_count.view(-1)
        flat_current = self._current_motion_bin_failed.view(-1)
        flat_index = ref_ids * self.motion_bin_failed_count.shape[1] + bin_ids
        ones = torch.ones_like(flat_index, dtype=torch.float32, device=self.device)
        flat_count.scatter_add_(0, flat_index, ones)
        flat_current.scatter_add_(0, flat_index, ones)
        self._refresh_motion_sampling_prob()

    def _sample_uniform_trajectory_phase_starts(self, ref_ids):
        demo_lengths = self.demo_lengths[ref_ids].clamp(min=1)
        sample_lengths = (demo_lengths - 1).clamp(min=1)
        phase = torch.floor(torch.rand(len(ref_ids), device=self.device) * sample_lengths.float()).long()
        return torch.minimum(phase, sample_lengths - 1)

    def _sample_mixed_phase_starts(self, ref_ids):
        phase = self._sample_uniform_trajectory_phase_starts(ref_ids)
        if len(ref_ids) == 0:
            return phase
        motion = self.cfg.motion
        zero_start_ratio = float(getattr(motion, "zero_start_ratio", 0.0))
        zero_mask = torch.rand(len(ref_ids), device=self.device) < zero_start_ratio
        sample_mask = ~zero_mask
        if bool(getattr(motion, "use_adaptive_phase_sampling", False)) and torch.any(sample_mask):
            sample_ref_ids = ref_ids[sample_mask]
            probs = self.motion_sampling_prob[sample_ref_ids]
            sampled_bins = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
            bin_start = sampled_bins * self.motion_bin_size_frames
            demo_lengths = self.demo_lengths[sample_ref_ids].clamp(min=1)
            max_start = (demo_lengths - 1).clamp(min=1)
            bin_end = torch.minimum(bin_start + self.motion_bin_size_frames, max_start)
            span = (bin_end - bin_start).clamp(min=1)
            offsets = torch.floor(torch.rand(len(sample_ref_ids), device=self.device) * span.float()).long()
            phase[sample_mask] = torch.minimum(bin_start + offsets, max_start - 1)
        phase[zero_mask] = 0
        return phase

    def post_physics_step(self):
        LeggedRobot.post_physics_step(self)

    def check_termination(self):
        termination_contact_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        grace_mask = self.episode_length_buf > 5
        self.contact_reset_buf = termination_contact_buf & grace_mask
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.base_too_low_buf = (self.root_states[:, 2] < 0.5) & grace_mask
        max_phase = self.demo_lengths[self.ref_idx].clamp(min=1) - 1
        self.ref_end_reset_buf = self.phase_idx >= max_phase

        term_cfg = getattr(self.cfg, "termination", None)
        if term_cfg is not None and bool(getattr(term_cfg, "use_tracking_error_termination", False)):
            tracking_grace = self.episode_length_buf > int(getattr(term_cfg, "tracking_termination_grace_steps", 5))
            ref_waist_z = self.ref_waist_pos[:, 0, 2]
            cur_waist_z = self.rigid_state[:, self.waist_body_id, 2]
            self.waist_z_bad_buf = (
                torch.abs(ref_waist_z - cur_waist_z) > float(getattr(term_cfg, "waist_z_threshold", 0.25))
            ) & tracking_grace
            ref_projected_gravity = quat_rotate_inverse(self.ref_waist_quat[:, 0, :], self.gravity_vec)
            cur_projected_gravity = quat_rotate_inverse(
                self.rigid_state[:, self.waist_body_id, 3:7],
                self.gravity_vec,
            )
            self.waist_ori_bad_buf = (
                torch.abs(ref_projected_gravity[:, 2] - cur_projected_gravity[:, 2])
                > float(getattr(term_cfg, "waist_ori_threshold", 0.8))
            ) & tracking_grace
            if len(self.feet_indices) == 2:
                cur_feet_z = self.rigid_state[:, self.feet_indices, 2]
                ref_feet_z = self.ref_feet_pos[:, :, 2]
                self.foot_z_bad_buf = (
                    torch.any(
                        torch.abs(cur_feet_z - ref_feet_z) > float(getattr(term_cfg, "foot_z_threshold", 0.25)),
                        dim=1,
                    )
                ) & tracking_grace
            else:
                self.foot_z_bad_buf[:] = False
            self.tracking_error_reset_buf = self.waist_z_bad_buf | self.waist_ori_bad_buf | self.foot_z_bad_buf
        else:
            self.waist_z_bad_buf[:] = False
            self.waist_ori_bad_buf[:] = False
            self.foot_z_bad_buf[:] = False
            self.tracking_error_reset_buf[:] = False

        self.fall_reset_buf = self.contact_reset_buf | self.base_too_low_buf | self.tracking_error_reset_buf
        self.adaptive_phase_failure_buf = self.fall_reset_buf & (~self.time_out_buf) & (~self.ref_end_reset_buf)
        self.reset_buf = self.fall_reset_buf | self.time_out_buf | self.ref_end_reset_buf

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
        self._update_adaptive_phase_failures(env_ids)
        LeggedRobot.reset_idx(self, env_ids)
        if len(env_ids) == 0:
            return
        episode_info = self.extras.get("episode", {})
        episode_info["fall_contact_ratio"] = torch.mean(self.contact_reset_buf[env_ids].float())
        episode_info["base_too_low_ratio"] = torch.mean(self.base_too_low_buf[env_ids].float())
        episode_info["tracking_error_ratio"] = torch.mean(self.tracking_error_reset_buf[env_ids].float())
        episode_info["waist_z_bad_ratio"] = torch.mean(self.waist_z_bad_buf[env_ids].float())
        episode_info["waist_ori_bad_ratio"] = torch.mean(self.waist_ori_bad_buf[env_ids].float())
        episode_info["foot_z_bad_ratio"] = torch.mean(self.foot_z_bad_buf[env_ids].float())
        episode_info["sampling_entropy"] = self.motion_sampling_entropy
        episode_info["sampling_top1_prob"] = self.motion_sampling_top1_prob
        episode_info["sampling_top1_bin"] = self.motion_sampling_top1_bin
        self.phase_idx[env_ids] = self.episode_phase_buf[env_ids]
        self.compute_ref_state()
        self._resample_ankle_obs_bias(env_ids)
        self._resample_ankle_dq_randomization(env_ids)
        self.initial_base_yaw[env_ids] = get_euler_xyz_tensor(self.root_states[env_ids, 3:7])[:, 2]
        self.contact_reset_buf[env_ids] = False
        self.base_too_low_buf[env_ids] = False
        self.tracking_error_reset_buf[env_ids] = False
        self.waist_z_bad_buf[env_ids] = False
        self.waist_ori_bad_buf[env_ids] = False
        self.foot_z_bad_buf[env_ids] = False
        self.adaptive_phase_failure_buf[env_ids] = False
