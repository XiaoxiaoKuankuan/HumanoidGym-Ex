from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.backends.isaaclab_backend import IsaacLabBackend
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_bpm_config_lab import MrobotMimicBPMLabCfg
from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_dance_config_lab import MrobotMimicDanceLabCfg
from humanoid_gym_ex.utils.mrobot_trajectory_reference import get_motion_files_from_cfg, load_mrobot_trajectory_library
from humanoid_gym_ex.utils.reference_state import JOINT_NAME_ALIASES, ReferenceStateNet

MrobotMimicCfg = MrobotMimicBPMLabCfg


def _quat_wxyz_to_xyzw(quat):
    return torch.cat((quat[..., 1:4], quat[..., 0:1]), dim=-1)


def _quat_xyzw_to_wxyz(quat):
    return torch.cat((quat[..., 3:4], quat[..., 0:3]), dim=-1)


def _quat_wxyz_to_euler_xyz(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _quat_rotate_inverse_wxyz(quat, vec):
    w = quat[:, 0]
    q_vec = quat[:, 1:4]
    a = vec * (2.0 * w * w - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, vec, dim=-1) * (-2.0 * w).unsqueeze(-1)
    c = q_vec * (2.0 * torch.sum(q_vec * vec, dim=-1, keepdim=True))
    return a + b + c


def _quat_apply_wxyz(quat, vec):
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    a = vec * (2.0 * w * w - 1.0)
    b = torch.cross(q_vec, vec, dim=-1) * (2.0 * w)
    c = q_vec * (2.0 * torch.sum(q_vec * vec, dim=-1, keepdim=True))
    return a + b + c


def _quat_conjugate_wxyz(quat):
    result = quat.clone()
    result[..., 1:4] = -result[..., 1:4]
    return result


def _quat_inv_wxyz(quat):
    return _quat_conjugate_wxyz(quat) / torch.sum(quat * quat, dim=-1, keepdim=True).clamp(min=1e-6)


def _quat_error_mag_wxyz(q1, q2):
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _quat_from_yaw_wxyz(yaw):
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    half = 0.5 * yaw
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat


def _calc_heading_quat_wxyz(quat):
    ref_dir = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    ref_dir[:, 0] = 1.0
    rot_dir = _quat_apply_wxyz(quat, ref_dir)
    heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
    return _quat_from_yaw_wxyz(heading)


def _quat_mul_wxyz(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _wrap_to_pi(angles):
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def _matrix_from_quat_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def _torch_rand_float_cpu(lower, upper, shape):
    return (float(upper) - float(lower)) * torch.rand(*shape, device="cpu") + float(lower)


def _resolve_reference_model_path(path):
    return path if os.path.isabs(path) else os.path.join(LEGGED_GYM_ROOT_DIR, path)


def _matched_control_decimation(mrobot_cfg_cls):
    """Return the control decimation, optionally matched to trajectory FPS."""
    decimation = int(getattr(mrobot_cfg_cls.control, "decimation", 1))
    if not bool(getattr(mrobot_cfg_cls.control, "match_reference_fps", False)):
        return max(1, decimation)
    reference_fps = float(getattr(mrobot_cfg_cls.motion, "reference_fps", 0.0))
    sim_dt = float(getattr(mrobot_cfg_cls.sim, "dt", 0.0))
    if reference_fps <= 0.0 or sim_dt <= 0.0:
        return max(1, decimation)
    raw_decimation = 1.0 / (reference_fps * sim_dt)
    matched = max(1, int(round(raw_decimation)))
    if not math.isclose(raw_decimation, float(matched), rel_tol=0.0, abs_tol=1e-6):
        print(
            "[HumanoidGym-Ex][WARN] MRobot reference FPS does not divide sim.dt exactly: "
            f"reference_fps={reference_fps}, sim_dt={sim_dt}, raw_decimation={raw_decimation:.6f}, "
            f"using decimation={matched}",
            flush=True,
        )
    # Keep the runtime mrobot cfg and DirectRLEnv cfg aligned.
    mrobot_cfg_cls.control.decimation = matched
    return matched


def _isaaclab_default_joint_angles(mrobot_cfg_cls=MrobotMimicBPMLabCfg):
    joint_angles = dict(mrobot_cfg_cls.init_state.default_joint_angles)
    # IsaacLab validates URDF limits during articulation initialization.  The
    # old IsaacGym default pose places these wrist roll joints just past the URDF
    # limits, so clamp only the IsaacLab initial pose and keep the shared mimic
    # config unchanged.
    joint_angles["upper_left_7_joint"] = -1.04
    joint_angles["upper_right_7_joint"] = 1.04
    return joint_angles


@configclass
class MrobotMimicIsaacLabEnvCfg(DirectRLEnvCfg):
    episode_length_s = MrobotMimicCfg.env.episode_length_s
    decimation = MrobotMimicCfg.control.decimation
    action_scale = MrobotMimicCfg.control.action_scale
    action_space = MrobotMimicCfg.env.num_policy_actions
    observation_space = MrobotMimicCfg.env.num_observations
    state_space = MrobotMimicCfg.env.num_privileged_obs
    reference_model_path = MrobotMimicCfg.motion.reference_model_path
    motion_files = None
    use_local_plane_terrain = True
    disable_domain_randomization = False
    deterministic_reset = False
    profile_step_timings = False
    profile_step_timing_interval = 200
    profile_step_timing_warmup = 20
    num_steps_per_env = None

    sim: SimulationCfg = SimulationCfg(
        dt=MrobotMimicCfg.sim.dt,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=MrobotMimicCfg.sim.physx.solver_type,
            min_position_iteration_count=MrobotMimicCfg.sim.physx.num_position_iterations,
            max_position_iteration_count=MrobotMimicCfg.sim.physx.num_position_iterations,
            min_velocity_iteration_count=MrobotMimicCfg.sim.physx.num_velocity_iterations,
            max_velocity_iteration_count=MrobotMimicCfg.sim.physx.num_velocity_iterations,
            bounce_threshold_velocity=MrobotMimicCfg.sim.physx.bounce_threshold_velocity,
            gpu_max_rigid_contact_count=MrobotMimicCfg.sim.physx.max_gpu_contact_pairs,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=MrobotMimicCfg.env.num_envs, env_spacing=3.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=MrobotMimicCfg.terrain.static_friction,
            dynamic_friction=MrobotMimicCfg.terrain.dynamic_friction,
            restitution=MrobotMimicCfg.terrain.restitution,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=MrobotMimicCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            fix_base=MrobotMimicCfg.asset.fix_base_link,
            merge_fixed_joints=True,
            activate_contact_sensors=True,
            self_collision=not bool(MrobotMimicCfg.asset.self_collisions),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=MrobotMimicCfg.sim.physx.max_depenetration_velocity,
                solver_position_iteration_count=MrobotMimicCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=MrobotMimicCfg.sim.physx.num_velocity_iterations,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(MrobotMimicCfg.asset.self_collisions),
                solver_position_iteration_count=MrobotMimicCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=MrobotMimicCfg.sim.physx.num_velocity_iterations,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(MrobotMimicCfg.init_state.pos),
            joint_pos=_isaaclab_default_joint_angles(),
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=MrobotMimicCfg.lab_joint_effort_limits,
                velocity_limit_sim=MrobotMimicCfg.lab_joint_velocity_limits,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    # Track only the IsaacGym termination body set.  Full-body contact sensing
    # is much slower with thousands of IsaacLab envs; name-based mapping below
    # keeps the subset order stable.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*(base_link|waist_yaw_link|pelvic_yaw_link|knee_pitch_link|ankle_roll_link)",
        history_length=1,
        update_period=MrobotMimicCfg.sim.dt * MrobotMimicCfg.control.decimation,
        track_air_time=False,
    )


@configclass
class MrobotMimicDanceIsaacLabEnvCfg(MrobotMimicIsaacLabEnvCfg):
    episode_length_s = MrobotMimicDanceLabCfg.env.episode_length_s
    decimation = _matched_control_decimation(MrobotMimicDanceLabCfg)
    action_scale = MrobotMimicDanceLabCfg.control.action_scale
    action_space = MrobotMimicDanceLabCfg.env.num_policy_actions
    observation_space = MrobotMimicDanceLabCfg.env.num_observations
    state_space = MrobotMimicDanceLabCfg.env.num_privileged_obs
    reference_model_path = ""
    motion_files = list(MrobotMimicDanceLabCfg.motion.files)
    sim: SimulationCfg = SimulationCfg(
        dt=MrobotMimicDanceLabCfg.sim.dt,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=MrobotMimicDanceLabCfg.sim.physx.solver_type,
            min_position_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_position_iterations,
            max_position_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_position_iterations,
            min_velocity_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_velocity_iterations,
            max_velocity_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_velocity_iterations,
            bounce_threshold_velocity=MrobotMimicDanceLabCfg.sim.physx.bounce_threshold_velocity,
            gpu_max_rigid_contact_count=MrobotMimicDanceLabCfg.sim.physx.max_gpu_contact_pairs,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=MrobotMimicDanceLabCfg.env.num_envs, env_spacing=3.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=MrobotMimicDanceLabCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            fix_base=MrobotMimicDanceLabCfg.asset.fix_base_link,
            merge_fixed_joints=True,
            activate_contact_sensors=True,
            self_collision=not bool(MrobotMimicDanceLabCfg.asset.self_collisions),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=MrobotMimicDanceLabCfg.sim.physx.max_depenetration_velocity,
                solver_position_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_velocity_iterations,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(MrobotMimicDanceLabCfg.asset.self_collisions),
                solver_position_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=MrobotMimicDanceLabCfg.sim.physx.num_velocity_iterations,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(MrobotMimicDanceLabCfg.init_state.pos),
            joint_pos=_isaaclab_default_joint_angles(MrobotMimicDanceLabCfg),
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=MrobotMimicDanceLabCfg.lab_joint_effort_limits,
                velocity_limit_sim=MrobotMimicDanceLabCfg.lab_joint_velocity_limits,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*(base_link|waist_yaw_link|pelvic_yaw_link|knee_pitch_link|ankle_roll_link)",
        history_length=1,
        update_period=MrobotMimicDanceLabCfg.sim.dt * decimation,
        track_air_time=False,
    )


class MrobotMimicIsaacLabEnv(DirectRLEnv):
    cfg: MrobotMimicIsaacLabEnvCfg
    mrobot_cfg_cls = MrobotMimicBPMLabCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.mrobot_cfg = self.mrobot_cfg_cls()
        if hasattr(cfg, "motion_files") and cfg.motion_files:
            self.mrobot_cfg.motion.files = list(cfg.motion_files)
        if getattr(cfg, "disable_domain_randomization", False):
            self.mrobot_cfg.domain_rand.randomize_friction = False
            self.mrobot_cfg.domain_rand.randomize_restitution = False
            self.mrobot_cfg.domain_rand.randomize_payload_mass = False
            self.mrobot_cfg.domain_rand.randomize_com_displacement = False
            self.mrobot_cfg.domain_rand.randomize_link_mass = False
            self.mrobot_cfg.domain_rand.push_robots = False
            self.mrobot_cfg.domain_rand.disturbance = False
            self.mrobot_cfg.domain_rand.randomize_kp = False
            self.mrobot_cfg.domain_rand.randomize_kd = False
            self.mrobot_cfg.domain_rand.randomize_motor_strength = False
            self.mrobot_cfg.domain_rand.randomize_motor_offset = False
            self.mrobot_cfg.domain_rand.randomize_default_dof_pos_offset = False
            self.mrobot_cfg.domain_rand.randomize_joint_friction = False
            self.mrobot_cfg.domain_rand.randomize_joint_armature = False
            self.mrobot_cfg.domain_rand.action_delay = False
        super().__init__(cfg, render_mode, **kwargs)
        self.backend = IsaacLabBackend(self, self.robot, self.contact_sensor)
        self.num_policy_actions = self.cfg.action_space
        self.num_actions = self.cfg.action_space
        self.num_obs = self.cfg.observation_space
        self.num_privileged_obs = self.cfg.state_space
        self._init_buffers()
        if self._uses_trajectory_reference():
            self._init_trajectory_library()
        else:
            self._init_reference_network()
        self._prepare_reward_function()
        self.update_domain_rand_curriculum(0, force=True)
        self._profile_step_count = 0
        self._profile_accum = {}
        self._active_profile_accum = None
        self._print_startup_diagnostics()

    def _profile_sync(self):
        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _profile_record(self, accum, name, seconds):
        if accum is not None:
            accum[name] = accum.get(name, 0.0) + float(seconds)

    def _profile_section_start(self):
        if not getattr(self.cfg, "profile_step_timings", False) or self._active_profile_accum is None:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_section_end(self, name, start):
        if start is None:
            return
        self._profile_sync()
        self._profile_record(self._active_profile_accum, name, time.perf_counter() - start)

    def _profile_report_step_timings(self, accum):
        self._profile_step_count += 1
        warmup = int(getattr(self.cfg, "profile_step_timing_warmup", 20))
        if self._profile_step_count <= warmup:
            return
        for name, value in accum.items():
            self._profile_accum[name] = self._profile_accum.get(name, 0.0) + value
        interval = max(1, int(getattr(self.cfg, "profile_step_timing_interval", 200)))
        profiled_steps = self._profile_step_count - warmup
        if profiled_steps % interval != 0:
            return
        ordered_names = [
            "pre_physics_step",
            "noncontrolled_ref_action",
            "apply_action",
            "action_filter",
            "action_delay",
            "write_data_to_sim",
            "sim_step",
            "render",
            "scene_update",
            "counters",
            "dones",
            "dones_phase_ref",
            "dones_state_cache",
            "dones_push_contact",
            "dones_contact_read",
            "dones_termination",
            "rewards",
            "reset",
            "reset_episode_logging",
            "reset_adaptive_sampling",
            "reset_robot_super",
            "reset_bpm_ref",
            "reset_domain_rand",
            "reset_state_write",
            "reset_cleanup",
            "events",
            "observations",
            "reference_update",
            "obs_noise",
            "state_root_joint",
            "state_base_vel",
            "state_body",
            "state_cache",
            "total",
        ]
        pieces = []
        for name in ordered_names:
            if name in self._profile_accum:
                pieces.append(f"{name}={self._profile_accum[name] * 1000.0 / interval:.3f}ms")
        print(
            "[MRobot IsaacLab profile] avg per env.step over "
            f"{interval} steps: " + ", ".join(pieces),
            flush=True,
        )
        num_steps_per_env = getattr(self.cfg, "num_steps_per_env", None)
        physics_substeps = None if num_steps_per_env is None else int(num_steps_per_env) * int(self.cfg.decimation)
        print(
            "[MRobot IsaacLab profile cfg] "
            f"num_steps_per_env={num_steps_per_env}, decimation={self.cfg.decimation}, "
            f"physics_substeps_per_rollout={physics_substeps}, policy_dt={self.step_dt:.6f}, "
            f"max_episode_length_steps={self.max_episode_length}",
            flush=True,
        )
        self._profile_accum.clear()

    def step(self, action):
        if not getattr(self.cfg, "profile_step_timings", False):
            return super().step(action)

        accum = {}
        self._active_profile_accum = accum
        self._profile_sync()
        total_start = last = time.perf_counter()

        def mark(name):
            nonlocal last
            self._profile_sync()
            now = time.perf_counter()
            self._profile_record(accum, name, now - last)
            last = now

        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)
        mark("pre_physics_step")

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            mark("apply_action")
            self.scene.write_data_to_sim()
            mark("write_data_to_sim")
            self.sim.step(render=False)
            mark("sim_step")
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
                mark("render")
            self.scene.update(dt=self.physics_dt)
            mark("scene_update")

        self.episode_length_buf += 1
        self.common_step_counter += 1
        mark("counters")

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        mark("dones")

        self.reward_buf = self._get_rewards()
        mark("rewards")

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        mark("reset")

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
        mark("events")

        self.obs_buf = self._get_observations()
        mark("observations")

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        mark("obs_noise")

        self._profile_sync()
        self._profile_record(accum, "total", time.perf_counter() - total_start)
        self._profile_report_step_timings(accum)
        self._active_profile_accum = None
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        use_local_plane = bool(self.cfg.use_local_plane_terrain and self.cfg.terrain.terrain_type == "plane")
        if use_local_plane:
            self._spawn_local_plane_terrain()
            self.terrain = None
        else:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene._terrain = self.terrain
        self.scene.clone_environments(copy_from_source=False)
        if use_local_plane:
            self.terrain = SimpleNamespace(env_origins=self.scene.env_origins, prim_path=self.cfg.terrain.prim_path)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_local_plane_terrain(self):
        """Spawn a local static ground plane without referencing IsaacSim's remote grid USD."""
        ground_cfg = sim_utils.CuboidCfg(
            size=(2000.0, 2000.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=self.cfg.terrain.physics_material,
        )
        ground_cfg.func(self.cfg.terrain.prim_path, ground_cfg, translation=(0.0, 0.0, -0.01))

    def _find_bodies(self, pattern):
        ids, _ = self.robot.find_bodies(pattern)
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _find_bodies_by_cfg_name(self, name):
        return self._find_bodies(".*{}.*".format(name))

    def _find_bodies_by_substrings(self, names):
        body_names = list(getattr(self.robot, "body_names", []) or [])
        ids = []
        for name in names:
            for body_id, body_name in enumerate(body_names):
                if name in body_name and body_id not in ids:
                    ids.append(body_id)
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _body_names_from_indices(self, indices):
        body_names = list(getattr(self.robot, "body_names", []) or [])
        result = []
        for idx in indices.detach().cpu().tolist():
            idx = int(idx)
            result.append(body_names[idx] if 0 <= idx < len(body_names) else f"<body:{idx}>")
        return result

    def _print_startup_diagnostics(self):
        source = getattr(getattr(self.mrobot_cfg, "motion", None), "reference_source", "bpm")
        print(
            "[HumanoidGym-Ex] MRobot IsaacLab startup: "
            f"task_reference={source}, num_envs={self.num_envs}, "
            f"action_space={self.cfg.action_space}, observation_space={self.cfg.observation_space}, "
            f"state_space={self.cfg.state_space}, device={self.device}",
            flush=True,
        )
        if source == "trajectory":
            print(
                "[HumanoidGym-Ex] Observation layout changed: Dance actor obs 61/73 -> 75, "
                "critic obs 210/222 -> 197. "
                "Old checkpoints and normalizer statistics are incompatible. "
                "Train from scratch or reset normalizer.",
                flush=True,
            )
            print(
                "[HumanoidGym-Ex] MRobot Dance actor q observation uses q - default_q. "
                "Old Dance checkpoints/normalizers from raw-q or q-ref-q observations are incompatible.",
                flush=True,
            )
            print(
                "[HumanoidGym-Ex] MRobot Dance motion files: "
                + ", ".join([str(path) for path in getattr(self, "motion_files", [])]),
                flush=True,
            )
            reference_fps = float(getattr(self.mrobot_cfg.motion, "reference_fps", 0.0))
            expected_reference_dt = 1.0 / reference_fps if reference_fps > 0.0 else float("nan")
            num_steps_per_env = getattr(self.cfg, "num_steps_per_env", None)
            physics_substeps = None if num_steps_per_env is None else int(num_steps_per_env) * int(self.cfg.decimation)
            print(
                "[HumanoidGym-Ex] MRobot Dance timing: "
                f"sim.dt={self.cfg.sim.dt}, control.decimation={self.cfg.decimation}, "
                f"policy_dt={self.step_dt:.6f}, reference_fps={reference_fps:g}, "
                f"expected_reference_dt={expected_reference_dt:.6f}, "
                f"physics_substeps_per_rollout={physics_substeps}, "
                f"max_episode_length_steps={self.max_episode_length}",
                flush=True,
            )
        else:
            print(
                "[HumanoidGym-Ex] Observation layout changed: actor obs 64 -> 76. "
                "Old checkpoints and normalizer statistics are incompatible. "
                "Train from scratch or reset normalizer.",
                flush=True,
            )
        print(
            "[HumanoidGym-Ex] MRobot body mapping: "
            f"base={self._body_names_from_indices(self.base_indices)}, "
            f"waist={self._body_names_from_indices(self.waist_indices)}, "
            f"feet={self._body_names_from_indices(self.feet_indices)}, "
            f"knee={self._body_names_from_indices(self.knee_indices)}, "
            f"hip={self._body_names_from_indices(self.hip_indices)}, "
            f"pelvic_yaw={self._body_names_from_indices(self.pelvic_yaw_indices)}",
            flush=True,
        )
        print(
            "[HumanoidGym-Ex] MRobot action preprocessing: "
            f"actions_filter={bool(getattr(self.mrobot_cfg.normalization, 'actions_filter', False))}, "
            f"action_delay={bool(getattr(self.mrobot_cfg.domain_rand, 'action_delay', False))}, "
            f"action_delay_range={getattr(self.mrobot_cfg.domain_rand, 'action_delay_range', None)}, "
            f"current_delay_ratio={float(getattr(self, '_current_delay_ratio', 0.0))}",
            flush=True,
        )
        print(
            "[HumanoidGym-Ex] MRobot tracking mapping: "
            f"anchor_body={self._body_names_from_indices(torch.as_tensor([self.waist_body_id], device=self.device))}, "
            f"tracking_body_count={len(self.all_tracking_indices)}, "
            f"tracking_body_names={self._body_names_from_indices(self.all_tracking_indices)}, "
            f"tracking_body_indices={self.all_tracking_indices.detach().cpu().tolist()}",
            flush=True,
        )
        sensor_body_names = list(getattr(self.contact_sensor, "body_names", []) or [])
        print(
            "[HumanoidGym-Ex] MRobot contact mapping: "
            f"termination_robot={self._body_names_from_indices(self.termination_robot_indices)}, "
            f"penalized_robot={self._body_names_from_indices(self.penalized_robot_indices)}, "
            f"termination_sensor_indices={self.termination_contact_indices.detach().cpu().tolist()}, "
            f"penalized_sensor_indices={self.penalised_contact_indices.detach().cpu().tolist()}, "
            f"feet_sensor_indices={self.feet_contact_indices.detach().cpu().tolist()}, "
            f"sensor_bodies={sensor_body_names}",
            flush=True,
        )

    def _body_index_from_name(self, name, default=0):
        ids, _ = self.robot.find_bodies(name)
        if len(ids) == 0:
            ids, _ = self.robot.find_bodies(".*{}.*".format(name))
        return int(ids[0]) if len(ids) else int(default)

    def _make_interval_steps(self, seconds_or_range):
        if isinstance(seconds_or_range, (list, tuple)):
            low = max(1, math.ceil(float(seconds_or_range[0]) / self.step_dt))
            high = max(low, math.ceil(float(seconds_or_range[1]) / self.step_dt))
            return (low, high), True
        steps = max(1, math.ceil(float(seconds_or_range) / self.step_dt))
        return steps, False

    def _sample_push_interval_steps(self):
        if self.randomize_push_interval:
            low, high = self.push_interval
            return int(torch.randint(low, high + 1, (1,), device=self.device).item())
        return int(self.push_interval)

    @staticmethod
    def _scale_one_center_range(target_range, ratio):
        low, high = float(target_range[0]), float(target_range[1])
        return [1.0 + (low - 1.0) * ratio, 1.0 + (high - 1.0) * ratio]

    @staticmethod
    def _scale_zero_center_range(target_range, ratio):
        return [float(target_range[0]) * ratio, float(target_range[1]) * ratio]

    @staticmethod
    def _scale_delay_range(target_range, ratio):
        low, high = int(target_range[0]), int(target_range[1])
        if ratio <= 0.0:
            return [0, 1]
        return [max(0, int(round(low * ratio))), max(1, int(round(high * ratio)))]

    def _init_domain_rand_curriculum_buffers(self):
        dr = self.mrobot_cfg.domain_rand
        self._domain_rand_curriculum_stage = -1
        self._target_push_robots = bool(getattr(dr, "push_robots", False))
        self._target_disturbance = bool(getattr(dr, "disturbance", False))
        self._target_randomize_restitution = bool(getattr(dr, "randomize_restitution", False))
        self._target_randomize_payload_mass = bool(getattr(dr, "randomize_payload_mass", False))
        self._target_randomize_com_displacement = bool(getattr(dr, "randomize_com_displacement", False))
        self._target_randomize_link_mass = bool(getattr(dr, "randomize_link_mass", False))
        self._target_randomize_kp = bool(getattr(dr, "randomize_kp", False))
        self._target_randomize_kd = bool(getattr(dr, "randomize_kd", False))
        self._target_randomize_motor_strength = bool(getattr(dr, "randomize_motor_strength", False))
        self._target_randomize_motor_offset = bool(getattr(dr, "randomize_motor_offset", False))
        self._target_action_delay = bool(getattr(dr, "action_delay", False))
        self._target_max_push_vel_xy = float(getattr(dr, "max_push_vel_xy", 0.0))
        self._target_max_push_ang_vel = float(getattr(dr, "max_push_ang_vel", 0.0))
        self._target_disturbance_range = list(getattr(dr, "disturbance_range", [0.0, 0.0]))
        self._target_restitution_range = list(getattr(dr, "restitution_range", [0.0, 0.0]))
        self._target_payload_mass_range = list(getattr(dr, "payload_mass_range", [0.0, 0.0]))
        self._target_com_x_pos_range = list(getattr(dr, "com_x_pos_range", [0.0, 0.0]))
        self._target_com_y_pos_range = list(getattr(dr, "com_y_pos_range", [0.0, 0.0]))
        self._target_com_z_pos_range = list(getattr(dr, "com_z_pos_range", [0.0, 0.0]))
        self._target_link_mass_range = list(getattr(dr, "link_mass_range", [1.0, 1.0]))
        self._target_kp_range = list(getattr(dr, "kp_range", [1.0, 1.0]))
        self._target_kd_range = list(getattr(dr, "kd_range", [1.0, 1.0]))
        self._target_motor_strength_range = list(getattr(dr, "motor_strength_range", [1.0, 1.0]))
        self._target_motor_offset_range = list(getattr(dr, "motor_offset_range", [0.0, 0.0]))
        self._target_action_delay_range = list(getattr(dr, "action_delay_range", [0, 1]))
        self._adaptive_curriculum_mean_episode_length = 0.0
        self._adaptive_curriculum_fall_ratio = 1.0
        self._adaptive_curriculum_resets = 0
        self._adaptive_curriculum_length_sum = torch.zeros((), device=self.device)
        self._adaptive_curriculum_fall_sum = torch.zeros((), device=self.device)
        self._adaptive_curriculum_pending_resets = 0
        self._adaptive_curriculum_current_iteration = 0
        self._adaptive_curriculum_stage_start_iteration = 0
        self._current_push_ratio = 0.0
        self._current_disturbance_ratio = 0.0
        self._current_restitution_ratio = 0.0
        self._current_payload_ratio = 0.0
        self._current_com_ratio = 0.0
        self._current_link_mass_ratio = 0.0
        self._current_pd_ratio = 0.0
        self._current_motor_strength_ratio = 0.0
        self._current_motor_offset_ratio = 0.0
        self._current_delay_ratio = 0.0

    def update_domain_rand_curriculum(self, iteration, force=False):
        dr = self.mrobot_cfg.domain_rand
        if not getattr(dr, "use_curriculum", False):
            return
        self._flush_adaptive_curriculum_metrics()
        self._adaptive_curriculum_current_iteration = iteration
        push_schedule = list(getattr(dr, "push_ratio_schedule", [1.0]))
        disturbance_schedule = list(getattr(dr, "disturbance_ratio_schedule", [1.0]))
        restitution_schedule = list(getattr(dr, "restitution_ratio_schedule", [1.0]))
        pd_schedule = list(getattr(dr, "pd_ratio_schedule", [1.0]))
        motor_strength_schedule = list(getattr(dr, "motor_strength_ratio_schedule", [1.0]))
        motor_offset_schedule = list(getattr(dr, "motor_offset_ratio_schedule", [1.0]))
        delay_schedule = list(getattr(dr, "delay_ratio_schedule", [1.0]))
        payload_schedule = list(getattr(dr, "payload_ratio_schedule", [1.0]))
        com_schedule = list(getattr(dr, "com_ratio_schedule", [1.0]))
        link_mass_schedule = list(getattr(dr, "link_mass_ratio_schedule", [1.0]))
        num_stages = max(
            len(push_schedule),
            len(disturbance_schedule),
            len(restitution_schedule),
            len(pd_schedule),
            len(motor_strength_schedule),
            len(motor_offset_schedule),
            len(delay_schedule),
            len(payload_schedule),
            len(com_schedule),
            len(link_mass_schedule),
        )
        if num_stages == 0:
            return
        curriculum_mode = getattr(dr, "curriculum_mode", "iteration")
        if curriculum_mode == "adaptive":
            current_stage = max(self._domain_rand_curriculum_stage, 0)
            stage_idx = current_stage
            if (
                iteration >= getattr(dr, "adaptive_min_iteration", 0)
                and self._adaptive_curriculum_resets >= getattr(dr, "adaptive_min_resets", 0)
                and (iteration - self._adaptive_curriculum_stage_start_iteration)
                >= getattr(dr, "adaptive_stage_cooldown_iterations", 0)
            ):
                length_thresholds = list(getattr(dr, "adaptive_length_ratio_thresholds", [0.0] * num_stages))
                fall_thresholds = list(getattr(dr, "adaptive_fall_ratio_thresholds", [1.0] * num_stages))
                candidate_stage = min(current_stage + 1, num_stages - 1)
                length_threshold = length_thresholds[min(candidate_stage, len(length_thresholds) - 1)]
                fall_threshold = fall_thresholds[min(candidate_stage, len(fall_thresholds) - 1)]
                mean_length_ratio = self._adaptive_curriculum_mean_episode_length / max(float(self.max_episode_length), 1.0)
                if mean_length_ratio >= length_threshold and self._adaptive_curriculum_fall_ratio <= fall_threshold:
                    stage_idx = candidate_stage
        else:
            stage_idx = 0
            for idx, start_iter in enumerate(list(getattr(dr, "curriculum_stage_iters", [0]))):
                if iteration >= start_iter:
                    stage_idx = idx
                else:
                    break
        if (not force) and stage_idx == self._domain_rand_curriculum_stage:
            return
        previous_stage = self._domain_rand_curriculum_stage
        push_ratio = push_schedule[min(stage_idx, len(push_schedule) - 1)]
        disturbance_ratio = disturbance_schedule[min(stage_idx, len(disturbance_schedule) - 1)]
        restitution_ratio = restitution_schedule[min(stage_idx, len(restitution_schedule) - 1)]
        pd_ratio = pd_schedule[min(stage_idx, len(pd_schedule) - 1)]
        motor_strength_ratio = motor_strength_schedule[min(stage_idx, len(motor_strength_schedule) - 1)]
        motor_offset_ratio = motor_offset_schedule[min(stage_idx, len(motor_offset_schedule) - 1)]
        delay_ratio = delay_schedule[min(stage_idx, len(delay_schedule) - 1)]
        payload_ratio = payload_schedule[min(stage_idx, len(payload_schedule) - 1)]
        com_ratio = com_schedule[min(stage_idx, len(com_schedule) - 1)]
        link_mass_ratio = link_mass_schedule[min(stage_idx, len(link_mass_schedule) - 1)]
        dr.push_robots = self._target_push_robots and push_ratio > 0.0
        dr.max_push_vel_xy = self._target_max_push_vel_xy * push_ratio
        dr.max_push_ang_vel = self._target_max_push_ang_vel * push_ratio
        dr.disturbance = self._target_disturbance and disturbance_ratio > 0.0
        dr.disturbance_range = self._scale_zero_center_range(self._target_disturbance_range, disturbance_ratio)
        dr.randomize_restitution = self._target_randomize_restitution and restitution_ratio > 0.0
        dr.restitution_range = self._scale_zero_center_range(self._target_restitution_range, restitution_ratio)
        dr.randomize_payload_mass = self._target_randomize_payload_mass and payload_ratio > 0.0
        dr.payload_mass_range = self._scale_zero_center_range(self._target_payload_mass_range, payload_ratio)
        dr.randomize_com_displacement = self._target_randomize_com_displacement and com_ratio > 0.0
        dr.com_x_pos_range = self._scale_zero_center_range(self._target_com_x_pos_range, com_ratio)
        dr.com_y_pos_range = self._scale_zero_center_range(self._target_com_y_pos_range, com_ratio)
        dr.com_z_pos_range = self._scale_zero_center_range(self._target_com_z_pos_range, com_ratio)
        dr.randomize_link_mass = self._target_randomize_link_mass and link_mass_ratio > 0.0
        dr.link_mass_range = self._scale_one_center_range(self._target_link_mass_range, link_mass_ratio)
        dr.randomize_kp = self._target_randomize_kp and pd_ratio > 0.0
        dr.kp_range = self._scale_one_center_range(self._target_kp_range, pd_ratio)
        dr.randomize_kd = self._target_randomize_kd and pd_ratio > 0.0
        dr.kd_range = self._scale_one_center_range(self._target_kd_range, pd_ratio)
        dr.randomize_motor_strength = self._target_randomize_motor_strength and motor_strength_ratio > 0.0
        dr.motor_strength_range = self._scale_one_center_range(self._target_motor_strength_range, motor_strength_ratio)
        dr.randomize_motor_offset = self._target_randomize_motor_offset and motor_offset_ratio > 0.0
        dr.motor_offset_range = self._scale_zero_center_range(self._target_motor_offset_range, motor_offset_ratio)
        dr.action_delay = self._target_action_delay and delay_ratio > 0.0
        dr.action_delay_range = (
            self._scale_delay_range(self._target_action_delay_range, delay_ratio)
            if self._target_action_delay
            else [0, 0]
        )
        self._domain_rand_curriculum_stage = stage_idx
        if curriculum_mode == "adaptive" and stage_idx != previous_stage:
            self._adaptive_curriculum_stage_start_iteration = iteration
            self._adaptive_curriculum_mean_episode_length = 0.0
            self._adaptive_curriculum_fall_ratio = 1.0
            self._adaptive_curriculum_resets = 0
            self._adaptive_curriculum_length_sum.zero_()
            self._adaptive_curriculum_fall_sum.zero_()
            self._adaptive_curriculum_pending_resets = 0
        self._current_push_ratio = push_ratio
        self._current_disturbance_ratio = disturbance_ratio
        self._current_restitution_ratio = restitution_ratio
        self._current_payload_ratio = payload_ratio
        self._current_com_ratio = com_ratio
        self._current_link_mass_ratio = link_mass_ratio
        self._current_pd_ratio = pd_ratio
        self._current_motor_strength_ratio = motor_strength_ratio
        self._current_motor_offset_ratio = motor_offset_ratio
        self._current_delay_ratio = delay_ratio if self._target_action_delay else 0.0
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_reset_buffers(all_env_ids)

    def _flush_adaptive_curriculum_metrics(self):
        if self._adaptive_curriculum_pending_resets <= 0:
            return
        pending_resets = self._adaptive_curriculum_pending_resets
        metric_sums = torch.stack(
            (self._adaptive_curriculum_length_sum, self._adaptive_curriculum_fall_sum)
        ).detach().cpu()
        batch_mean_episode_length = float(metric_sums[0].item()) / max(float(pending_resets), 1.0)
        fall_ratio = float(metric_sums[1].item()) / max(float(pending_resets), 1.0)
        dr = self.mrobot_cfg.domain_rand
        ema = getattr(dr, "adaptive_metric_ema", 0.9)
        if self._adaptive_curriculum_resets == 0:
            self._adaptive_curriculum_mean_episode_length = batch_mean_episode_length
            self._adaptive_curriculum_fall_ratio = fall_ratio
        else:
            self._adaptive_curriculum_mean_episode_length = (
                ema * self._adaptive_curriculum_mean_episode_length + (1.0 - ema) * batch_mean_episode_length
            )
            self._adaptive_curriculum_fall_ratio = ema * self._adaptive_curriculum_fall_ratio + (1.0 - ema) * fall_ratio
        self._adaptive_curriculum_resets += pending_resets
        self._adaptive_curriculum_length_sum.zero_()
        self._adaptive_curriculum_fall_sum.zero_()
        self._adaptive_curriculum_pending_resets = 0

    def _canonical_to_sim_order(self, canonical_values):
        sim_values = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        sim_values[:, self.joint_sim_ids] = canonical_values
        return sim_values

    def _init_buffers(self):
        cfg = self.mrobot_cfg
        self.num_control = list(cfg.env.num_control)
        self.num_notcontrol = list(cfg.env.num_notcontrol)
        self.ref_num_notcontrol = list(cfg.env.ref_num_notcontrol)
        self.dof_err_w = torch.tensor(
            getattr(cfg.rewards, "dof_err_w", [1.0] * len(self.num_control)),
            dtype=torch.float32,
            device=self.device,
        )
        if self.dof_err_w.numel() != len(self.num_control):
            raise RuntimeError(
                "MRobot IsaacLab rewards.dof_err_w length mismatch: "
                f"got {self.dof_err_w.numel()}, expected {len(self.num_control)}"
            )
        self.canonical_joint_names = list(cfg.init_state.default_joint_angles.keys())
        joint_sim_ids = [self.robot.joint_names.index(name) for name in self.canonical_joint_names]
        self.joint_sim_ids = torch.tensor(joint_sim_ids, dtype=torch.long, device=self.device)
        self.joint_sim_ids_list = [int(idx) for idx in joint_sim_ids]
        self.joint_sim_ids_cpu = self.joint_sim_ids.detach().cpu()
        self.full_actions = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.last_full_actions = torch.zeros_like(self.full_actions)
        self.delayed_full_actions_scaled = torch.zeros_like(self.full_actions)
        self.target_dof_pos = torch.zeros_like(self.full_actions)
        self.sim_order_torques = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_policy_actions, device=self.device)
        self.env_ids_arange = torch.arange(self.num_envs, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.joint_sim_ids].clone()
        self.dof_pos_limits = self.robot.data.joint_pos_limits[:, self.joint_sim_ids].clone()
        self.p_gains = torch.zeros(cfg.env.num_actions, device=self.device)
        self.d_gains = torch.zeros(cfg.env.num_actions, device=self.device)
        for i, name in enumerate(self.canonical_joint_names):
            for key, value in cfg.control.stiffness.items():
                if key in name:
                    self.p_gains[i] = value
            for key, value in cfg.control.damping.items():
                if key in name:
                    self.d_gains[i] = value
        torque_limits = torch.tensor(
            [float(cfg.lab_joint_effort_limits[name]) for name in self.canonical_joint_names],
            device=self.device,
        )
        torque_limits = torch.where(
            torch.isfinite(torque_limits) & (torque_limits > 0.0),
            torque_limits,
            torch.ones_like(torque_limits) * 250.0,
        )
        self.torque_limits = torque_limits * cfg.safety.torque_limit
        self.obs_scales = cfg.normalization.obs_scales
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.bpm_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self.init_phase_rad = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase_rad = torch.zeros(self.num_envs, 1, device=self.device)
        self.normalized_bpm_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self.ref_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.data_length = 1
        self.demo_length = max(1, int(round(self.max_episode_length_s / self.step_dt)))
        self.demo_lengths = torch.full((1,), self.demo_length, dtype=torch.long, device=self.device)
        self.ref_dof_pos = self.default_dof_pos.clone()
        self.ref_dof_vel = torch.zeros_like(self.ref_dof_pos)
        self.ref_pelvis_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_pelvis_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_pelvis_quat = self._identity_quat(1)
        self.ref_pelvis_ang_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_feet_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_feet_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_feet_quat = self._identity_quat(2)
        self.ref_feet_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_knee_quat = self._identity_quat(2)
        self.ref_knee_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_hip_quat = self._identity_quat(2)
        self.ref_hip_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_pelvic_yaw_quat = self._identity_quat(2)
        self.ref_pelvic_yaw_ang_vel = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.ref_waist_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_waist_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ref_waist_quat = self._identity_quat(1)
        self.ref_waist_ang_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.tracking_ref_pos_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.tracking_ref_quat_buf = self._identity_quat(1)
        self.tracking_ref_lin_vel_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.tracking_ref_ang_vel_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.priv_tracking_ref_pos_buf = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self.priv_tracking_ref_quat_buf = self._identity_quat(10)
        self.priv_tracking_ref_lin_vel_buf = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self.priv_tracking_ref_ang_vel_buf = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self.ref_feet_contact = torch.zeros(self.num_envs, 2, device=self.device)
        self.ref_foot_height = torch.zeros(self.num_envs, 2, device=self.device)
        self._use_feet_pos_z_for_ref_foot_height = False
        self.ref_root_linvel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_angvel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_euler_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        # Match IsaacGym's body indexing semantics: resolve key bodies from
        # cfg.asset substring names once at startup.  The contact sensor prim
        # path remains a Lab-only performance filter, but all runtime body sets
        # come from the same config names used by Gym.
        self.feet_indices = self._find_bodies_by_cfg_name(cfg.asset.foot_name)
        self.knee_indices = self._find_bodies_by_cfg_name(cfg.asset.knee_name)
        self.ankle_indices = self._find_bodies_by_cfg_name(cfg.asset.ankle_name)
        self.hip_indices = self._find_bodies_by_cfg_name(cfg.asset.hip_name)
        self.pelvic_yaw_indices = self._find_bodies_by_cfg_name(getattr(cfg.asset, "pelvic_yaw_name", "pelvic_yaw_link"))
        self.base_indices = self._find_bodies_by_cfg_name(cfg.asset.base_name)
        self.waist_indices = self._find_bodies_by_cfg_name(cfg.asset.waist_name)
        self.base_body_id = int(self.base_indices[0].item()) if len(self.base_indices) else 0
        self.waist_body_id = int(self.waist_indices[0].item()) if len(self.waist_indices) else self.base_body_id
        self.priv_tracking_indices = torch.cat(
            (self.base_indices, self.feet_indices, self.knee_indices, self.hip_indices, self.pelvic_yaw_indices, self.waist_indices),
            dim=0,
        ).long()
        self._priv_tracking_pos_splits = (3, 6, 6, 6, 6, 3)
        self._priv_tracking_quat_splits = (4, 8, 8, 8, 8, 4)
        self.all_tracking_indices = self._resolve_reward_tracking_body_indices(getattr(cfg.asset, "tracking_body_names", None))
        self.tracking_body_names = self._body_names_from_indices(self.all_tracking_indices)
        self._tracking_ref_specs = self._make_tracking_ref_specs(self.all_tracking_indices)
        self._priv_tracking_ref_specs = self._make_tracking_ref_specs(self.priv_tracking_indices)
        self.tracking_ref_pos_buf = torch.zeros(self.num_envs, len(self.all_tracking_indices), 3, device=self.device)
        self.tracking_ref_quat_buf = self._identity_quat(len(self.all_tracking_indices))
        self.tracking_ref_lin_vel_buf = torch.zeros(self.num_envs, len(self.all_tracking_indices), 3, device=self.device)
        self.tracking_ref_ang_vel_buf = torch.zeros(self.num_envs, len(self.all_tracking_indices), 3, device=self.device)
        self._validate_tracking_body_indices()
        termination_robot_indices = self._find_bodies_by_substrings(getattr(cfg.asset, "terminate_after_contacts_on", []))
        if len(termination_robot_indices) == 0:
            termination_robot_indices = torch.cat(
                (self.base_indices, self.waist_indices, self.pelvic_yaw_indices, self.knee_indices), dim=0
            ).long()
        self.termination_robot_indices = termination_robot_indices.long()
        self.termination_contact_indices = self._contact_sensor_indices_for_robot_bodies(self.termination_robot_indices)
        self.feet_contact_indices = self._contact_sensor_indices_for_robot_bodies(self.feet_indices)
        if len(self.termination_contact_indices) != len(self.termination_robot_indices):
            sensor_names = ", ".join(getattr(self.contact_sensor, "body_names", []) or [])
            robot_names = ", ".join(getattr(self.robot, "body_names", []) or [])
            raise RuntimeError(
                "MRobot IsaacLab termination contact mapping mismatch: "
                f"sensor matched {len(self.termination_contact_indices)} bodies, "
                f"expected {len(self.termination_robot_indices)}. "
                f"contact_sensor.body_names=[{sensor_names}], robot.body_names=[{robot_names}]"
            )
        penalized_robot_indices = self._find_bodies_by_substrings(getattr(cfg.asset, "penalize_contacts_on", []))
        if len(penalized_robot_indices) == 0:
            penalized_robot_indices = self.termination_robot_indices
        self.penalized_robot_indices = penalized_robot_indices.long()
        self.penalised_contact_indices = self._contact_sensor_indices_for_robot_bodies(self.penalized_robot_indices)
        self._tracking_cache_valid = False
        self._tracking_cache_common_step = -1
        self.contact_forces = self.backend.get_contact_forces()
        self.rigid_state = self.robot.data.body_state_w
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        self.base_quat = self.robot.data.root_quat_w
        self.base_euler_xyz = _quat_wxyz_to_euler_xyz(self.base_quat)
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.initial_base_yaw = torch.zeros(self.num_envs, device=self.device)
        self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        self.disturbance_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.disturbance_torque = torch.zeros_like(self.disturbance_force)
        self.external_force_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._external_force_active_any = False
        self.payload = torch.zeros(self.num_envs, 1, device=self.device)
        self.com_displacement = torch.zeros(self.num_envs, 3, device=self.device)
        self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device) * cfg.terrain.static_friction
        self.dynamic_friction_coeffs = torch.ones(self.num_envs, 1, device=self.device) * cfg.terrain.dynamic_friction
        self.restitution_coeffs = torch.ones(self.num_envs, 1, device=self.device) * cfg.terrain.restitution
        self.Kp_factors = torch.ones(self.num_envs, cfg.env.num_actions, device=self.device)
        self.Kd_factors = torch.ones(self.num_envs, cfg.env.num_actions, device=self.device)
        self.motor_strength_factors = torch.ones(self.num_envs, cfg.env.num_actions, device=self.device)
        self.motor_offsets = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.default_dof_pos_offsets = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.common_step_counter = 0
        self.joint_armature_coeffs = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.joint_friction_coeffs = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self._joint_armature_written_once = False
        self._joint_friction_written_once = False
        self.default_masses = self.robot.data.default_mass.detach().cpu().clone()
        self.default_inertias = self.robot.data.default_inertia.detach().cpu().clone()
        self.default_coms = self.robot.root_physx_view.get_coms().clone().cpu()
        self._physx_masses_cpu = self.default_masses.clone()
        self._physx_inertias_cpu = self.default_inertias.clone()
        self._physx_coms_cpu = self.default_coms.clone()
        self._physx_materials_cpu = None
        try:
            self._physx_materials_cpu = self.robot.root_physx_view.get_material_properties().clone()
        except Exception as exc:
            if not getattr(self, "_reported_material_cache_error", False):
                print("[HumanoidGym-Ex] MRobot IsaacLab material cache unavailable:", exc, flush=True)
                self._reported_material_cache_error = True
        self.payload_body_id = self._body_index_from_name(getattr(cfg.domain_rand, "payload_body_name", cfg.asset.base_name), default=0)
        self.com_body_id = self._body_index_from_name(getattr(cfg.domain_rand, "com_body_name", cfg.asset.base_name), default=0)
        self.disturbance_body_id = self._body_index_from_name(cfg.asset.base_name, default=0)
        self.action_delay_buffer = None
        self.action_delay_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_delay_write_idx = 0
        self.action_delay_buffer_size = 0
        if getattr(cfg.domain_rand, "action_delay", False):
            # Delay is counted in physics substeps because the buffer is
            # advanced from _apply_action(), which IsaacLab calls once per
            # sim step inside each policy/control step.  Delay seconds are
            # therefore action_delay_timestep * cfg.sim.dt.
            max_delay = max(0, int(getattr(cfg.domain_rand, "action_delay_range", [0, 1])[1]))
            # Keep one extra slot so a delay equal to max_delay does not alias
            # to zero delay in the circular buffer.
            self.action_delay_buffer_size = max_delay + 1
            self.action_delay_buffer = torch.zeros(
                self.num_envs,
                cfg.env.num_actions,
                self.action_delay_buffer_size,
                device=self.device,
            )
        self._ankle_obs_joint_indices = torch.tensor(
            getattr(cfg.domain_rand, "ankle_obs_joint_indices", [4, 5, 10, 11]),
            dtype=torch.long,
            device=self.device,
        )
        self.ankle_reward_indices = torch.tensor([4, 5, 10, 11], dtype=torch.long, device=self.device)
        n_ankle_obs = len(self._ankle_obs_joint_indices)
        self.ankle_obs_pos_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device)
        self.ankle_obs_vel_bias = torch.zeros(self.num_envs, n_ankle_obs, device=self.device)
        dr = cfg.domain_rand
        self._use_actor_ankle_obs_randomization = any(
            bool(getattr(dr, name, False))
            for name in (
                "randomize_ankle_obs_pos_bias",
                "randomize_ankle_obs_vel_bias",
                "randomize_ankle_obs_vel_noise",
                "randomize_ankle_obs_vel_delay",
                "randomize_ankle_obs_vel_filter",
            )
        )
        self._use_ankle_pd_dq_randomization = any(
            bool(getattr(dr, name, False))
            for name in (
                "randomize_ankle_pd_dq_noise",
                "randomize_ankle_pd_dq_delay",
                "randomize_ankle_pd_dq_filter",
            )
        )
        self._init_ankle_dq_randomization_buffers(n_ankle_obs)
        self._init_sys_delay_buffers()
        self.push_interval, self.randomize_push_interval = self._make_interval_steps(cfg.domain_rand.push_interval_s)
        self.next_push_step = self.common_step_counter + self._sample_push_interval_steps()
        self.disturbance_interval = max(1, math.ceil(float(cfg.domain_rand.disturbance_s) / self.step_dt))
        self.fall_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_reset_buf = torch.zeros_like(self.fall_reset_buf)
        self.base_too_low_buf = torch.zeros_like(self.fall_reset_buf)
        self.ref_end_reset_buf = torch.zeros_like(self.fall_reset_buf)
        self.tracking_error_reset_buf = torch.zeros_like(self.fall_reset_buf)
        self.waist_z_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.waist_ori_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.foot_z_bad_buf = torch.zeros_like(self.fall_reset_buf)
        self.adaptive_phase_failure_buf = torch.zeros_like(self.fall_reset_buf)
        self.curriculum_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._init_domain_rand_curriculum_buffers()
        self._apply_substep = 0
        self.last_torques = torch.zeros_like(self.torques)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.policy_obs_buf = torch.zeros(self.num_envs, cfg.env.num_observations, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, cfg.env.num_privileged_obs, device=self.device)
        self.priv_curr_dim = int(cfg.env.num_privileged_obs - cfg.env.single_num_privileged_obs - cfg.env.num_goal_obs)
        self.priv_curr_buf = torch.zeros(self.num_envs, self.priv_curr_dim, device=self.device)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.motion_bin_size_frames = 1
        self.motion_num_bins = torch.ones(1, dtype=torch.long, device=self.device)
        self.motion_valid_bin_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        self.motion_bin_failed_count = torch.zeros(1, 1, device=self.device)
        self._current_motion_bin_failed = torch.zeros_like(self.motion_bin_failed_count)
        self.motion_sampling_prob = torch.ones_like(self.motion_bin_failed_count)
        self.motion_sampling_entropy = torch.ones((), device=self.device)
        self.motion_sampling_top1_prob = torch.ones((), device=self.device)
        self.motion_sampling_top1_bin = torch.zeros((), device=self.device)
        self._state_cache_common_step = -1
        self._state_cache_valid = False

    def _validate_tracking_body_indices(self):
        expected = {
            "base_link": (self.base_indices, 1),
            "waist_yaw_link": (self.waist_indices, 1),
            "ankle_roll feet": (self.feet_indices, 2),
            "knee_pitch": (self.knee_indices, 2),
            "pelvic_roll hip": (self.hip_indices, 2),
            "pelvic_yaw": (self.pelvic_yaw_indices, 2),
        }
        bad = []
        for label, (indices, count) in expected.items():
            if len(indices) != count:
                bad.append(f"{label}: got {len(indices)}, expected {count}")
        configured_tracking = list(getattr(self.mrobot_cfg.asset, "tracking_body_names", []) or [])
        if configured_tracking and len(self.all_tracking_indices) != len(configured_tracking):
            bad.append(
                f"tracking_body_names: got {len(self.all_tracking_indices)}, "
                f"expected {len(configured_tracking)}"
            )
        if len(torch.unique(self.all_tracking_indices)) != len(self.all_tracking_indices):
            bad.append("tracking_body_names resolved to duplicate body indices")
        if bad:
            body_names = ", ".join(getattr(self.robot, "body_names", []))
            raise RuntimeError(
                "MRobot IsaacLab body mapping mismatch. "
                + "; ".join(bad)
                + ". Imported body_names=["
                + body_names
                + "]"
            )

    def _resolve_reward_tracking_body_indices(self, configured_names):
        if not configured_names:
            return self.priv_tracking_indices.clone()
        body_names = list(getattr(self.robot, "body_names", []) or [])
        ids = []
        missing = []
        for name in configured_names:
            body_id = None
            if name in body_names:
                body_id = body_names.index(name)
            else:
                matches = [idx for idx, body_name in enumerate(body_names) if str(name) in body_name]
                if len(matches) == 1:
                    body_id = matches[0]
                elif len(matches) > 1:
                    missing.append(f"{name}: matched multiple bodies {[body_names[idx] for idx in matches]}")
                else:
                    missing.append(str(name))
            if body_id is not None:
                ids.append(body_id)
        if missing:
            raise RuntimeError(
                "MRobot IsaacLab tracking_body_names could not be resolved uniquely: "
                + "; ".join(missing)
                + ". Imported body_names=["
                + ", ".join(body_names)
                + "]"
            )
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    @staticmethod
    def _index_in_tensor(indices, body_id):
        matches = (indices == int(body_id)).nonzero(as_tuple=False).flatten()
        if len(matches) == 0:
            return None
        return int(matches[0].item())

    def _tracking_ref_spec_for_body_id(self, body_id):
        body_id = int(body_id)
        part_idx = self._index_in_tensor(self.waist_indices, body_id)
        if part_idx is not None:
            return ("waist", part_idx)
        part_idx = self._index_in_tensor(self.feet_indices, body_id)
        if part_idx is not None:
            return ("feet", part_idx)
        part_idx = self._index_in_tensor(self.knee_indices, body_id)
        if part_idx is not None:
            return ("knee", part_idx)
        part_idx = self._index_in_tensor(self.hip_indices, body_id)
        if part_idx is not None:
            return ("hip", part_idx)
        part_idx = self._index_in_tensor(self.pelvic_yaw_indices, body_id)
        if part_idx is not None:
            return ("pelvic_yaw", part_idx)
        part_idx = self._index_in_tensor(self.base_indices, body_id)
        if part_idx is not None:
            # The legacy privileged keypoint block uses base_link as the
            # simulated pelvis/root body and compares it to pelvis reference.
            return ("pelvis", 0)
        body_names = list(getattr(self.robot, "body_names", []) or [])
        body_name = body_names[body_id] if 0 <= body_id < len(body_names) else f"<body:{body_id}>"
        raise RuntimeError(f"No reference keypoint mapping exists for tracking body {body_name}.")

    def _make_tracking_ref_specs(self, indices):
        return [self._tracking_ref_spec_for_body_id(body_id) for body_id in indices.detach().cpu().tolist()]

    def _contact_sensor_indices_for_robot_bodies(self, robot_body_ids):
        sensor_body_names = list(getattr(self.contact_sensor, "body_names", []) or [])
        if len(sensor_body_names) == 0:
            return torch.zeros(1, dtype=torch.long, device=self.device)
        robot_body_names = list(getattr(self.robot, "body_names", []) or [])
        ids = []
        for robot_body_id in robot_body_ids.detach().cpu().tolist():
            if 0 <= int(robot_body_id) < len(robot_body_names):
                body_name = robot_body_names[int(robot_body_id)]
                if body_name in sensor_body_names:
                    ids.append(sensor_body_names.index(body_name))
        if not ids and len(sensor_body_names) == 1:
            ids = [0]
        if not ids:
            ids = [0]
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _identity_quat(self, num_parts):
        quat = torch.zeros(self.num_envs, num_parts, 4, device=self.device)
        quat[..., 0] = 1.0
        return quat

    def _get_noise_scale_vec(self):
        cfg = self.mrobot_cfg
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        noise_scales = cfg.noise.noise_scales
        n_ctrl = len(self.num_control)
        noise_vec[0:n_ctrl] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[n_ctrl:2 * n_ctrl] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[2 * n_ctrl:3 * n_ctrl] = 0.0
        noise_vec[3 * n_ctrl:3 * n_ctrl + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[3 * n_ctrl + 3:3 * n_ctrl + 6] = noise_scales.euler
        return noise_vec

    def _resample_ankle_obs_bias(self, env_ids):
        if len(env_ids) == 0:
            return
        dr = self.mrobot_cfg.domain_rand
        n = len(self._ankle_obs_joint_indices)
        if getattr(dr, "randomize_ankle_obs_pos_bias", False):
            low, high = getattr(dr, "ankle_obs_pos_bias_range", [-0.02, 0.02])
            self.ankle_obs_pos_bias[env_ids] = _torch_rand_float(low, high, (len(env_ids), n), self.device)
        else:
            self.ankle_obs_pos_bias[env_ids] = 0.0
        if getattr(dr, "randomize_ankle_obs_vel_bias", False):
            low, high = getattr(dr, "ankle_obs_vel_bias_range", [-0.3, 0.3])
            self.ankle_obs_vel_bias[env_ids] = _torch_rand_float(low, high, (len(env_ids), n), self.device)
        else:
            self.ankle_obs_vel_bias[env_ids] = 0.0

    def _init_ankle_dq_randomization_buffers(self, n_ankle_obs):
        dr = self.mrobot_cfg.domain_rand
        obs_delay_range = getattr(dr, "ankle_obs_vel_delay_range", [0, 0])
        pd_delay_range = getattr(dr, "ankle_pd_dq_delay_range", [0, 0])
        max_obs_delay = int(max(obs_delay_range)) if len(obs_delay_range) > 0 else 0
        max_pd_delay = int(max(pd_delay_range)) if len(pd_delay_range) > 0 else 0
        self.ankle_obs_vel_delay_buffer = torch.zeros(
            self.num_envs, n_ankle_obs, max(1, max_obs_delay + 1), device=self.device
        )
        self.ankle_obs_vel_delay_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ankle_obs_vel_filter_alpha = torch.ones(self.num_envs, n_ankle_obs, device=self.device)
        self.ankle_obs_vel_filtered = torch.zeros(self.num_envs, n_ankle_obs, device=self.device)
        self.ankle_obs_vel_filter_initialized = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.ankle_pd_dq_delay_buffer = torch.zeros(
            self.num_envs, n_ankle_obs, max(1, max_pd_delay + 1), device=self.device
        )
        self.ankle_pd_dq_delay_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ankle_pd_dq_filter_alpha = torch.ones(self.num_envs, n_ankle_obs, device=self.device)
        self.ankle_pd_dq_filtered = torch.zeros(self.num_envs, n_ankle_obs, device=self.device)
        self.ankle_pd_dq_filter_initialized = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def _sample_ankle_filter_alpha(self, env_ids, cutoff_range):
        cutoff = _torch_rand_float(
            float(cutoff_range[0]),
            float(cutoff_range[1]),
            (len(env_ids), len(self._ankle_obs_joint_indices)),
            self.device,
        ).clamp(min=0.0)
        return (1.0 - torch.exp(-2.0 * math.pi * cutoff * self.step_dt)).clamp(0.0, 1.0)

    def _resample_ankle_dq_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
        dr = self.mrobot_cfg.domain_rand
        if getattr(dr, "randomize_ankle_obs_vel_delay", False):
            low, high = getattr(dr, "ankle_obs_vel_delay_range", [0, 0])
            self.ankle_obs_vel_delay_timestep[env_ids] = torch.randint(
                int(low), int(high) + 1, (len(env_ids),), device=self.device
            )
        else:
            self.ankle_obs_vel_delay_timestep[env_ids] = 0
        if getattr(dr, "randomize_ankle_obs_vel_filter", False):
            cutoff_range = getattr(dr, "ankle_obs_vel_filter_cutoff_range", [8.0, 20.0])
            self.ankle_obs_vel_filter_alpha[env_ids] = self._sample_ankle_filter_alpha(env_ids, cutoff_range)
        else:
            self.ankle_obs_vel_filter_alpha[env_ids] = 1.0
        self.ankle_obs_vel_delay_buffer[env_ids] = 0.0
        self.ankle_obs_vel_filtered[env_ids] = 0.0
        self.ankle_obs_vel_filter_initialized[env_ids] = False

        if getattr(dr, "randomize_ankle_pd_dq_delay", False):
            low, high = getattr(dr, "ankle_pd_dq_delay_range", [0, 0])
            self.ankle_pd_dq_delay_timestep[env_ids] = torch.randint(
                int(low), int(high) + 1, (len(env_ids),), device=self.device
            )
        else:
            self.ankle_pd_dq_delay_timestep[env_ids] = 0
        if getattr(dr, "randomize_ankle_pd_dq_filter", False):
            cutoff_range = getattr(dr, "ankle_pd_dq_filter_cutoff_range", [8.0, 20.0])
            self.ankle_pd_dq_filter_alpha[env_ids] = self._sample_ankle_filter_alpha(env_ids, cutoff_range)
        else:
            self.ankle_pd_dq_filter_alpha[env_ids] = 1.0
        self.ankle_pd_dq_delay_buffer[env_ids] = 0.0
        self.ankle_pd_dq_filtered[env_ids] = 0.0
        self.ankle_pd_dq_filter_initialized[env_ids] = False

    def _delay_ankle_signal(self, signal, delay_buffer, delay_timestep):
        if delay_buffer.shape[-1] <= 1:
            delay_buffer[:, :, 0] = signal
            return signal
        delay_buffer[:, :, 1:] = delay_buffer[:, :, :-1].clone()
        delay_buffer[:, :, 0] = signal
        return delay_buffer[self.env_ids_arange, :, delay_timestep.long()]

    def _filter_ankle_signal(self, signal, filtered, alpha, initialized):
        filtered[:] = torch.where(initialized, (1.0 - alpha) * filtered + alpha * signal, signal)
        initialized[:] = True
        return filtered

    def _get_ankle_dq_for_pd(self):
        if not self._use_ankle_pd_dq_randomization:
            return self.dof_vel
        dof_vel_for_pd = self.dof_vel.clone()
        ankle_dq = self.dof_vel[:, self._ankle_obs_joint_indices]
        dr = self.mrobot_cfg.domain_rand
        if getattr(dr, "randomize_ankle_pd_dq_delay", False):
            ankle_dq = self._delay_ankle_signal(ankle_dq, self.ankle_pd_dq_delay_buffer, self.ankle_pd_dq_delay_timestep)
        if getattr(dr, "randomize_ankle_pd_dq_filter", False):
            ankle_dq = self._filter_ankle_signal(
                ankle_dq,
                self.ankle_pd_dq_filtered,
                self.ankle_pd_dq_filter_alpha,
                self.ankle_pd_dq_filter_initialized,
            )
        if getattr(dr, "randomize_ankle_pd_dq_noise", False):
            ankle_dq = ankle_dq + torch.randn_like(ankle_dq) * float(getattr(dr, "ankle_pd_dq_noise_std", 0.0))
        dof_vel_for_pd[:, self._ankle_obs_joint_indices] = ankle_dq
        return dof_vel_for_pd

    def _apply_actor_ankle_obs_bias(self, q, dq):
        if not self._use_actor_ankle_obs_randomization:
            return q, dq
        q_actor = q.clone()
        dq_actor = dq.clone()
        ankle_dq = dq_actor[:, self._ankle_obs_joint_indices]
        dr = self.mrobot_cfg.domain_rand
        if getattr(dr, "randomize_ankle_obs_vel_delay", False):
            ankle_dq = self._delay_ankle_signal(ankle_dq, self.ankle_obs_vel_delay_buffer, self.ankle_obs_vel_delay_timestep)
        if getattr(dr, "randomize_ankle_obs_vel_filter", False):
            ankle_dq = self._filter_ankle_signal(
                ankle_dq,
                self.ankle_obs_vel_filtered,
                self.ankle_obs_vel_filter_alpha,
                self.ankle_obs_vel_filter_initialized,
            )
        if getattr(dr, "randomize_ankle_obs_vel_noise", False):
            ankle_dq = ankle_dq + torch.randn_like(ankle_dq) * float(getattr(dr, "ankle_obs_vel_noise_std", 0.0)) * self.obs_scales.dof_vel
        q_actor[:, self._ankle_obs_joint_indices] += self.ankle_obs_pos_bias * self.obs_scales.dof_pos
        dq_actor[:, self._ankle_obs_joint_indices] = ankle_dq + self.ankle_obs_vel_bias * self.obs_scales.dof_vel
        return q_actor, dq_actor

    def _init_sys_delay_buffers(self):
        dr = self.mrobot_cfg.domain_rand
        self.obs_imu_delay_buffer = None
        self.obs_motor_delay_buffer = None
        self.obs_imu_delay_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_motor_delay_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if getattr(dr, "sys_delay", False):
            imu_high = max(1, int(getattr(dr, "imu_delay_range", [0, 1])[1]))
            motor_high = max(1, int(getattr(dr, "motor_delay_range", [0, 1])[1]))
            self.obs_imu_delay_buffer = torch.zeros(self.num_envs, 13, imu_high, device=self.device)
            self.obs_motor_delay_buffer = torch.zeros(self.num_envs, self.mrobot_cfg.env.num_actions * 2, motor_high, device=self.device)

    def _resample_sys_delay(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        if not getattr(dr, "sys_delay", False) or self.obs_imu_delay_buffer is None:
            self.obs_imu_delay_timestep[env_ids] = 0
            self.obs_motor_delay_timestep[env_ids] = 0
            return
        self.obs_imu_delay_buffer[env_ids] = 0.0
        self.obs_motor_delay_buffer[env_ids] = 0.0
        imu_low, imu_high = getattr(dr, "imu_delay_range", [0, 1])
        motor_low, motor_high = getattr(dr, "motor_delay_range", [0, 1])
        imu_low_i = int(imu_low)
        imu_high_i = max(imu_low_i + 1, int(imu_high))
        motor_low_i = int(motor_low)
        motor_high_i = max(motor_low_i + 1, int(motor_high))
        self.obs_imu_delay_timestep[env_ids] = torch.randint(imu_low_i, imu_high_i, (len(env_ids),), device=self.device)
        self.obs_motor_delay_timestep[env_ids] = torch.randint(motor_low_i, motor_high_i, (len(env_ids),), device=self.device)

    def _record_sys_delay_state(self):
        if self.obs_imu_delay_buffer is None:
            return
        self.obs_imu_delay_buffer[:, :, 1:] = self.obs_imu_delay_buffer[:, :, :-1].clone()
        self.obs_imu_delay_buffer[:, :, 0] = self.robot.data.root_state_w.clone()
        self.obs_motor_delay_buffer[:, :, 1:] = self.obs_motor_delay_buffer[:, :, :-1].clone()
        self.obs_motor_delay_buffer[:, :, 0] = torch.cat((self.dof_pos, self.dof_vel), dim=1).clone()

    def _prime_sys_delay_state(self, env_ids, root_state=None, dof_pos=None, dof_vel=None):
        if self.obs_imu_delay_buffer is None:
            return
        if root_state is None:
            root_state = self.robot.data.root_state_w[env_ids].clone()
        if dof_pos is None:
            dof_pos = self.dof_pos[env_ids].clone()
        if dof_vel is None:
            dof_vel = self.dof_vel[env_ids].clone()
        motor_state = torch.cat((dof_pos, dof_vel), dim=1).clone()
        self.obs_imu_delay_buffer[env_ids] = root_state.unsqueeze(-1).expand(-1, -1, self.obs_imu_delay_buffer.shape[-1])
        self.obs_motor_delay_buffer[env_ids] = motor_state.unsqueeze(-1).expand(-1, -1, self.obs_motor_delay_buffer.shape[-1])

    def _get_sys_delayed_obs_state(self):
        root_states = self.obs_imu_delay_buffer[self.env_ids_arange, :, self.obs_imu_delay_timestep.long()]
        dof_pos_vel = self.obs_motor_delay_buffer[self.env_ids_arange, :, self.obs_motor_delay_timestep.long()]
        return root_states, dof_pos_vel

    def _randomize_reset_buffers(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        n_envs = len(env_ids)
        if getattr(dr, "randomize_kp", False):
            self.Kp_factors[env_ids] = _torch_rand_float(dr.kp_range[0], dr.kp_range[1], (n_envs, self.mrobot_cfg.env.num_actions), self.device)
        else:
            self.Kp_factors[env_ids] = 1.0
        if getattr(dr, "randomize_ankle_pd", False):
            ankle_kp = getattr(dr, "ankle_kp_range", None)
            ankle_idx = getattr(dr, "ankle_joint_indices", None)
            if ankle_kp is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kp_factors[env_ids, idx] = _torch_rand_float(ankle_kp[0], ankle_kp[1], (n_envs,), self.device)
        if getattr(dr, "randomize_kd", False):
            self.Kd_factors[env_ids] = _torch_rand_float(dr.kd_range[0], dr.kd_range[1], (n_envs, self.mrobot_cfg.env.num_actions), self.device)
        else:
            self.Kd_factors[env_ids] = 1.0
        if getattr(dr, "randomize_ankle_pd", False):
            ankle_kd = getattr(dr, "ankle_kd_range", None)
            ankle_idx = getattr(dr, "ankle_joint_indices", None)
            if ankle_kd is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.Kd_factors[env_ids, idx] = _torch_rand_float(ankle_kd[0], ankle_kd[1], (n_envs,), self.device)
        if getattr(dr, "randomize_motor_strength", False):
            self.motor_strength_factors[env_ids] = _torch_rand_float(
                dr.motor_strength_range[0], dr.motor_strength_range[1], (n_envs, self.mrobot_cfg.env.num_actions), self.device
            )
        else:
            self.motor_strength_factors[env_ids] = 1.0
        if getattr(dr, "randomize_motor_offset", False):
            self.motor_offsets[env_ids] = _torch_rand_float(
                dr.motor_offset_range[0], dr.motor_offset_range[1], (n_envs, self.mrobot_cfg.env.num_actions), self.device
            )
        else:
            self.motor_offsets[env_ids] = 0.0
        if getattr(dr, "randomize_ankle_motor_offset", False):
            ankle_offset = getattr(dr, "ankle_motor_offset_range", None)
            ankle_idx = getattr(dr, "ankle_joint_indices", None)
            if ankle_offset is not None and ankle_idx is not None:
                for idx in ankle_idx:
                    self.motor_offsets[env_ids, idx] = _torch_rand_float(ankle_offset[0], ankle_offset[1], (n_envs,), self.device)
        if getattr(dr, "randomize_default_dof_pos_offset", False):
            offset_range = getattr(dr, "default_dof_pos_offset_range", [-0.01, 0.01])
            self.default_dof_pos_offsets[env_ids] = _torch_rand_float(
                offset_range[0], offset_range[1], (n_envs, self.mrobot_cfg.env.num_actions), self.device
            )
            ankle_range = getattr(dr, "default_dof_pos_offset_ankle_range", None)
            if ankle_range is not None:
                for idx in getattr(dr, "default_dof_pos_offset_ankle_indices", []):
                    self.default_dof_pos_offsets[env_ids, idx] = _torch_rand_float(
                        ankle_range[0], ankle_range[1], (n_envs,), self.device
                    )
        else:
            self.default_dof_pos_offsets[env_ids] = 0.0
        if self.action_delay_buffer is not None:
            self.action_delay_buffer[env_ids] = 0.0
            if getattr(dr, "action_delay", False):
                low, high = getattr(dr, "action_delay_range", [0, 1])
                low_i = max(0, int(low))
                # torch.randint uses an exclusive high bound.  Config ranges
                # are treated as inclusive physics-substep delays.
                high_i = max(low_i, int(high)) + 1
                self.action_delay_timestep[env_ids] = torch.randint(low_i, high_i, (n_envs,), device=self.device)
            else:
                self.action_delay_timestep[env_ids] = 0
        self._resample_ankle_obs_bias(env_ids)
        self._resample_ankle_dq_randomization(env_ids)
        self._resample_sys_delay(env_ids)
        if self._should_resample_physx_randomization(env_ids):
            self._randomize_materials(env_ids)
            self._randomize_mass_and_com(env_ids)
            self._randomize_joint_physx_props(env_ids)

    def _should_resample_physx_randomization(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        if getattr(dr, "resample_physx_randomization_on_small_reset", True):
            return True
        return len(env_ids) == self.num_envs

    def _randomize_materials(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        if not (getattr(dr, "randomize_friction", False) or getattr(dr, "randomize_restitution", False)):
            self.friction_coeffs[env_ids] = self.mrobot_cfg.terrain.static_friction
            self.dynamic_friction_coeffs[env_ids] = self.mrobot_cfg.terrain.dynamic_friction
            self.restitution_coeffs[env_ids] = self.mrobot_cfg.terrain.restitution
            return
        env_ids_cpu = env_ids.detach().cpu()
        n_envs = len(env_ids)
        static_range = getattr(dr, "static_friction_range", getattr(dr, "friction_range", [1.0, 1.0]))
        dynamic_range = getattr(dr, "dynamic_friction_range", getattr(dr, "friction_range", [1.0, 1.0]))
        restitution_range = getattr(dr, "restitution_range", [0.0, 0.0])
        static_cpu = _torch_rand_float_cpu(static_range[0], static_range[1], (n_envs, 1))
        dynamic_cpu = _torch_rand_float_cpu(dynamic_range[0], dynamic_range[1], (n_envs, 1))
        restitution_cpu = _torch_rand_float_cpu(restitution_range[0], restitution_range[1], (n_envs, 1))
        if not getattr(dr, "randomize_friction", False):
            static_cpu[:] = self.mrobot_cfg.terrain.static_friction
            dynamic_cpu[:] = self.mrobot_cfg.terrain.dynamic_friction
        if not getattr(dr, "randomize_restitution", False):
            restitution_cpu[:] = self.mrobot_cfg.terrain.restitution
        self.friction_coeffs[env_ids] = static_cpu.to(self.device)
        self.dynamic_friction_coeffs[env_ids] = dynamic_cpu.to(self.device)
        self.restitution_coeffs[env_ids] = restitution_cpu.to(self.device)
        try:
            if self._physx_materials_cpu is None:
                self._physx_materials_cpu = self.robot.root_physx_view.get_material_properties().clone()
            material_samples = self._physx_materials_cpu[env_ids_cpu].clone()
            material_samples[..., 0] = static_cpu.expand_as(material_samples[..., 0])
            material_samples[..., 1] = dynamic_cpu.expand_as(material_samples[..., 1])
            material_samples[..., 2] = restitution_cpu.expand_as(material_samples[..., 2])
            self._physx_materials_cpu[env_ids_cpu] = material_samples
            self.robot.root_physx_view.set_material_properties(self._physx_materials_cpu, env_ids_cpu)
        except Exception as exc:
            if not getattr(self, "_reported_material_randomization_error", False):
                print("[HumanoidGym-Ex] MRobot IsaacLab material randomization kept in buffers only:", exc, flush=True)
                self._reported_material_randomization_error = True

    def _randomize_mass_and_com(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        env_ids_cpu = env_ids.detach().cpu()
        n_envs = len(env_ids)
        self._physx_masses_cpu[env_ids_cpu] = self.default_masses[env_ids_cpu]
        self._physx_inertias_cpu[env_ids_cpu] = self.default_inertias[env_ids_cpu]
        if getattr(dr, "randomize_link_mass", False):
            low, high = getattr(dr, "link_mass_range", [1.0, 1.0])
            scale = _torch_rand_float_cpu(low, high, (n_envs, self.robot.num_bodies))
            scale[:, self.payload_body_id] = 1.0
            self._physx_masses_cpu[env_ids_cpu] = self.default_masses[env_ids_cpu] * scale
            self._physx_inertias_cpu[env_ids_cpu] = self.default_inertias[env_ids_cpu] * scale.unsqueeze(-1)
        if getattr(dr, "randomize_payload_mass", False):
            low, high = getattr(dr, "payload_mass_range", [0.0, 0.0])
            payload_cpu = _torch_rand_float_cpu(low, high, (n_envs, 1))
        else:
            payload_cpu = torch.zeros(n_envs, 1, device="cpu")
        self.payload[env_ids] = payload_cpu.to(self.device)
        base_payload_mass = self._physx_masses_cpu[env_ids_cpu, self.payload_body_id].clone()
        new_payload_mass = torch.clamp(base_payload_mass + payload_cpu.squeeze(-1), min=1e-6)
        ratio = new_payload_mass / torch.clamp(base_payload_mass, min=1e-6)
        self._physx_masses_cpu[env_ids_cpu, self.payload_body_id] = new_payload_mass
        self._physx_inertias_cpu[env_ids_cpu, self.payload_body_id] = (
            self._physx_inertias_cpu[env_ids_cpu, self.payload_body_id] * ratio.unsqueeze(-1)
        )
        try:
            self.robot.root_physx_view.set_masses(self._physx_masses_cpu, env_ids_cpu)
            self.robot.root_physx_view.set_inertias(self._physx_inertias_cpu, env_ids_cpu)
        except Exception as exc:
            if not getattr(self, "_reported_mass_randomization_error", False):
                print("[HumanoidGym-Ex] MRobot IsaacLab mass randomization kept in buffers only:", exc, flush=True)
                self._reported_mass_randomization_error = True
        com_offset_cpu = torch.tensor(
            [
                float(getattr(dr, "com_offset_x", 0.0)),
                float(getattr(dr, "com_offset_y", 0.0)),
                float(getattr(dr, "com_offset_z", 0.0)),
            ],
            device="cpu",
        ).repeat(n_envs, 1)
        if getattr(dr, "randomize_com_displacement", False):
            com_offset_cpu[:, 0] += _torch_rand_float_cpu(dr.com_x_pos_range[0], dr.com_x_pos_range[1], (n_envs,))
            com_offset_cpu[:, 1] += _torch_rand_float_cpu(dr.com_y_pos_range[0], dr.com_y_pos_range[1], (n_envs,))
            com_offset_cpu[:, 2] += _torch_rand_float_cpu(dr.com_z_pos_range[0], dr.com_z_pos_range[1], (n_envs,))
        self.com_displacement[env_ids] = com_offset_cpu.to(self.device)
        try:
            self._physx_coms_cpu[env_ids_cpu] = self.default_coms[env_ids_cpu]
            self._physx_coms_cpu[env_ids_cpu, self.com_body_id, :3] = (
                self.default_coms[env_ids_cpu, self.com_body_id, :3] + com_offset_cpu
            )
            self.robot.root_physx_view.set_coms(self._physx_coms_cpu, env_ids_cpu)
        except Exception as exc:
            if not getattr(self, "_reported_com_randomization_error", False):
                print("[HumanoidGym-Ex] MRobot IsaacLab COM randomization kept in buffers only:", exc, flush=True)
                self._reported_com_randomization_error = True

    def _randomize_joint_physx_props(self, env_ids):
        dr = self.mrobot_cfg.domain_rand
        n_envs = len(env_ids)
        randomize_armature = bool(getattr(dr, "randomize_joint_armature", False))
        randomize_friction = bool(getattr(dr, "randomize_joint_friction", False))
        if randomize_armature:
            for i, rng in enumerate(getattr(dr, "joint_armature_range", [])):
                self.joint_armature_coeffs[env_ids, i] = _torch_rand_float(rng[0], rng[1], (n_envs,), self.device)
        else:
            values = getattr(dr, "joint_armature_values", None)
            if values is not None:
                for i, value in enumerate(values[: self.mrobot_cfg.env.num_actions]):
                    self.joint_armature_coeffs[env_ids, i] = float(value)
            else:
                self.joint_armature_coeffs[env_ids] = 0.0
        if randomize_friction:
            for i, rng in enumerate(getattr(dr, "joint_friction_range", [])):
                self.joint_friction_coeffs[env_ids, i] = _torch_rand_float(rng[0], rng[1], (n_envs,), self.device)
        else:
            self.joint_friction_coeffs[env_ids] = 0.0
        should_write_armature = randomize_armature or not self._joint_armature_written_once
        should_write_friction = randomize_friction or not self._joint_friction_written_once
        if not (should_write_armature or should_write_friction):
            return
        try:
            if should_write_armature:
                self.robot.write_joint_armature_to_sim(
                    self.joint_armature_coeffs[env_ids],
                    joint_ids=self.joint_sim_ids_list,
                    env_ids=env_ids,
                )
                if not randomize_armature:
                    self._joint_armature_written_once = True
            if should_write_friction:
                self.robot.write_joint_friction_coefficient_to_sim(
                    self.joint_friction_coeffs[env_ids],
                    joint_ids=self.joint_sim_ids_list,
                    env_ids=env_ids,
                )
                if not randomize_friction:
                    self._joint_friction_written_once = True
        except Exception as exc:
            if not getattr(self, "_reported_joint_prop_randomization_error", False):
                print("[HumanoidGym-Ex] MRobot IsaacLab joint property randomization kept in buffers only:", exc, flush=True)
                self._reported_joint_prop_randomization_error = True

    def _init_reference_network(self):
        path = _resolve_reference_model_path(getattr(self.cfg, "reference_model_path", self.mrobot_cfg.motion.reference_model_path))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "BPM reference model checkpoint not found: {}. Set env_cfg.reference_model_path; "
                "datasets and checkpoints are intentionally not bundled.".format(path)
            )
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.reference_output_columns = list(checkpoint["output_columns"])
        self.reference_column_index = {name: idx for idx, name in enumerate(self.reference_output_columns)}
        self.reference_bpm_mean = torch.tensor(float(checkpoint["bpm_mean"]), device=self.device)
        self.reference_bpm_std = torch.tensor(float(checkpoint["bpm_std"]), device=self.device).clamp(min=1e-6)
        self.reference_target_mean = torch.as_tensor(checkpoint["target_mean"], device=self.device, dtype=torch.float32)
        self.reference_target_std = torch.as_tensor(checkpoint["target_std"], device=self.device, dtype=torch.float32)
        self.reference_net = ReferenceStateNet(int(checkpoint["input_dim"]), int(checkpoint["output_dim"]), checkpoint["hidden"]).to(self.device)
        self.reference_net.load_state_dict(checkpoint["model_state_dict"])
        self.reference_net.eval()
        for param in self.reference_net.parameters():
            param.requires_grad_(False)
        self.reference_input = torch.zeros(
            self.num_envs,
            int(checkpoint["input_dim"]),
            device=self.device,
            dtype=torch.float32,
        )
        self.ref_dof_pos_indices, self.ref_dof_pos_mask = self._build_dof_column_indices("_pos")
        self.ref_dof_vel_indices, self.ref_dof_vel_mask = self._build_dof_column_indices("_vel")

    def _uses_trajectory_reference(self):
        return getattr(getattr(self.mrobot_cfg, "motion", None), "reference_source", "bpm") == "trajectory"

    def _init_trajectory_library(self):
        motion_cfg = getattr(self.mrobot_cfg, "motion", None)
        motion_files = get_motion_files_from_cfg(motion_cfg)
        library = load_mrobot_trajectory_library(
            motion_files,
            self.device,
            allow_legacy_keypoint_fallback=bool(getattr(motion_cfg, "allow_legacy_keypoint_fallback", False)),
            foot_contact_height_threshold=float(getattr(motion_cfg, "foot_contact_height_threshold", 0.08)),
        )
        self.motion_files = library.files
        self.feet_contact_source = getattr(library, "feet_contact_source", "unknown")
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
        self.pelvis_quat_buffer = _quat_xyzw_to_wxyz(buffers["pelvis_quat"])
        self.pelvis_ang_vel_buffer = buffers["pelvis_ang_vel"]
        self.feet_pos_buffer = buffers["feet_pos"]
        self.feet_vel_buffer = buffers["feet_vel"]
        self.feet_quat_buffer = _quat_xyzw_to_wxyz(buffers["feet_quat"])
        self.feet_ang_vel_buffer = buffers["feet_ang_vel"]
        self.knee_pos_buffer = buffers["knee_pos"]
        self.knee_vel_buffer = buffers["knee_vel"]
        self.knee_quat_buffer = _quat_xyzw_to_wxyz(buffers["knee_quat"])
        self.knee_ang_vel_buffer = buffers["knee_ang_vel"]
        self.hip_pos_buffer = buffers["hip_pos"]
        self.hip_vel_buffer = buffers["hip_vel"]
        self.hip_quat_buffer = _quat_xyzw_to_wxyz(buffers["hip_quat"])
        self.hip_ang_vel_buffer = buffers["hip_ang_vel"]
        self.pelvic_yaw_pos_buffer = buffers["pelvic_yaw_pos"]
        self.pelvic_yaw_vel_buffer = buffers["pelvic_yaw_vel"]
        self.pelvic_yaw_quat_buffer = _quat_xyzw_to_wxyz(buffers["pelvic_yaw_quat"])
        self.pelvic_yaw_ang_vel_buffer = buffers["pelvic_yaw_ang_vel"]
        self.waist_pos_buffer = buffers["waist_pos"]
        self.waist_vel_buffer = buffers["waist_vel"]
        self.waist_quat_buffer = _quat_xyzw_to_wxyz(buffers["waist_quat"])
        self.waist_ang_vel_buffer = buffers["waist_ang_vel"]
        self._check_trajectory_foot_height_semantics(getattr(library, "optional_buffers", {}))
        self._init_adaptive_phase_sampling()
        print("[mrobot_dance][IsaacLab] 指定轨迹 reference 加载完成", flush=True)
        print(f"[mrobot_dance][IsaacLab] 轨迹数量: {self.data_length}", flush=True)
        print(
            f"[mrobot_dance][IsaacLab] 各轨迹真实长度: {[int(x) for x in self.demo_lengths.detach().cpu().tolist()]}",
            flush=True,
        )
        print(
            "[mrobot_dance][IsaacLab] adaptive phase sampling: "
            f"enabled={getattr(motion_cfg, 'use_adaptive_phase_sampling', False)}, "
            f"reference_fps={getattr(motion_cfg, 'reference_fps', None)}, "
            f"bin_size_frames={self.motion_bin_size_frames}, "
            f"num_bins={[int(x) for x in self.motion_num_bins.detach().cpu().tolist()]}",
            flush=True,
        )
        print(f"[mrobot_dance][IsaacLab] feet_contact_source={self.feet_contact_source}", flush=True)

    def _check_trajectory_foot_height_semantics(self, optional_buffers=None):
        foot_height = self.foot_height_buffer
        feet_pos_z = self.feet_pos_buffer[..., 2]
        diff = torch.abs(foot_height - feet_pos_z)
        mean_abs = float(diff.mean().item())
        max_abs = float(diff.max().item())
        self._use_feet_pos_z_for_ref_foot_height = mean_abs > 1e-4
        message = (
            "[mrobot_dance][IsaacLab] foot_height check: "
            f"mean_abs_diff(foot_height, feet_pos[...,2])={mean_abs:.8f}, "
            f"max_abs_diff={max_abs:.8f}, "
            f"foot_height_minmax=({float(foot_height.min().item()):.6f}, {float(foot_height.max().item()):.6f}), "
            f"feet_pos_z_minmax=({float(feet_pos_z.min().item()):.6f}, {float(feet_pos_z.max().item()):.6f})"
        )
        optional_buffers = optional_buffers or {}
        for key in ("ground_height", "env_origin_z"):
            value = optional_buffers.get(key)
            if value is not None:
                message += f", {key}_minmax=({float(value.min().item()):.6f}, {float(value.max().item()):.6f})"
        print(message, flush=True)
        if self._use_feet_pos_z_for_ref_foot_height:
            print(
                "[mrobot_dance][IsaacLab] WARNING: foot_height and feet_pos[...,2] differ. "
                "Using ref_feet_pos[...,2] as ref_foot_height so foot-height reward compares world z to world z.",
                flush=True,
            )
        else:
            print(
                "[mrobot_dance][IsaacLab] foot_height matches feet_pos[...,2]; "
                "foot-height reward compares world z to world z.",
                flush=True,
            )

    def _build_dof_column_indices(self, suffix):
        reverse_alias = {env_name: data_name for data_name, env_name in JOINT_NAME_ALIASES.items()}
        indices, mask = [], []
        for dof_name in self.canonical_joint_names:
            base = reverse_alias.get(dof_name, dof_name)
            if base.endswith("_joint"):
                base = base[:-6]
            col_idx = self.reference_column_index.get(base + suffix, -1)
            indices.append(max(col_idx, 0))
            mask.append(col_idx >= 0)
        return (
            torch.tensor(indices, dtype=torch.long, device=self.device),
            torch.tensor(mask, dtype=torch.bool, device=self.device),
        )

    def _extract_ref_field(self, pred, field_name, num_parts, width, kind):
        axes = ("x", "y", "z") if width == 3 else ("x", "y", "z", "w")
        values = torch.zeros(self.num_envs, num_parts, width, device=self.device)
        if width == 4:
            values[..., 3] = 1.0
        for part_idx in range(num_parts):
            for axis_idx, axis in enumerate(axes):
                idx = self.reference_column_index.get(f"{field_name}_{kind}_{part_idx}_{axis}")
                if idx is not None:
                    values[:, part_idx, axis_idx] = pred[:, idx]
        if width == 4:
            values = _quat_xyzw_to_wxyz(values)
            values = values / values.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return values

    def compute_ref_state(self):
        if self._uses_trajectory_reference():
            self._compute_trajectory_ref_state()
            return
        self.phase_rad = torch.remainder(self.phase_rad, 2.0 * math.pi)
        self.reference_input[:, 0:1] = (self.bpm_cmd - self.reference_bpm_mean) / self.reference_bpm_std
        self.reference_input[:, 1:2] = torch.sin(self.phase_rad)
        self.reference_input[:, 2:3] = torch.cos(self.phase_rad)
        with torch.inference_mode():
            pred = self.reference_net(self.reference_input) * self.reference_target_std + self.reference_target_mean
        self.ref_dof_pos[:] = self.default_dof_pos
        dof_pred = pred[:, self.ref_dof_pos_indices]
        self.ref_dof_pos[:, self.ref_dof_pos_mask] = dof_pred[:, self.ref_dof_pos_mask]
        self.ref_dof_vel.zero_()
        dof_vel_pred = pred[:, self.ref_dof_vel_indices]
        self.ref_dof_vel[:, self.ref_dof_vel_mask] = dof_vel_pred[:, self.ref_dof_vel_mask]
        self.ref_pelvis_pos[:] = self._extract_ref_field(pred, "pelvis", 1, 3, "pos")
        self.ref_pelvis_vel[:] = self._extract_ref_field(pred, "pelvis", 1, 3, "vel")
        self.ref_pelvis_quat[:] = self._extract_ref_field(pred, "pelvis", 1, 4, "quat")
        self.ref_pelvis_ang_vel[:] = self._extract_ref_field(pred, "pelvis", 1, 3, "ang_vel")
        self.ref_feet_pos[:] = self._extract_ref_field(pred, "feet", 2, 3, "pos")
        self.ref_feet_vel[:] = self._extract_ref_field(pred, "feet", 2, 3, "vel")
        self.ref_feet_quat[:] = self._extract_ref_field(pred, "feet", 2, 4, "quat")
        self.ref_feet_ang_vel[:] = self._extract_ref_field(pred, "feet", 2, 3, "ang_vel")
        self.ref_knee_pos[:] = self._extract_ref_field(pred, "knee", 2, 3, "pos")
        self.ref_knee_vel[:] = self._extract_ref_field(pred, "knee", 2, 3, "vel")
        self.ref_knee_quat[:] = self._extract_ref_field(pred, "knee", 2, 4, "quat")
        self.ref_knee_ang_vel[:] = self._extract_ref_field(pred, "knee", 2, 3, "ang_vel")
        self.ref_hip_pos[:] = self._extract_ref_field(pred, "hip", 2, 3, "pos")
        self.ref_hip_vel[:] = self._extract_ref_field(pred, "hip", 2, 3, "vel")
        self.ref_hip_quat[:] = self._extract_ref_field(pred, "hip", 2, 4, "quat")
        self.ref_hip_ang_vel[:] = self._extract_ref_field(pred, "hip", 2, 3, "ang_vel")
        self.ref_pelvic_yaw_pos[:] = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "pos")
        self.ref_pelvic_yaw_vel[:] = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "vel")
        self.ref_pelvic_yaw_quat[:] = self._extract_ref_field(pred, "pelvic_yaw", 2, 4, "quat")
        self.ref_pelvic_yaw_ang_vel[:] = self._extract_ref_field(pred, "pelvic_yaw", 2, 3, "ang_vel")
        self.ref_waist_pos[:] = self._extract_ref_field(pred, "waist", 1, 3, "pos")
        self.ref_waist_vel[:] = self._extract_ref_field(pred, "waist", 1, 3, "vel")
        self.ref_waist_quat[:] = self._extract_ref_field(pred, "waist", 1, 4, "quat")
        self.ref_waist_ang_vel[:] = self._extract_ref_field(pred, "waist", 1, 3, "ang_vel")
        self.ref_root_linvel = self.ref_waist_vel[:, 0, :]
        self.ref_root_angvel = self.ref_waist_ang_vel[:, 0, :]
        self.ref_foot_height[:] = self.ref_feet_pos[..., 2]
        self.ref_feet_contact = (self.ref_feet_pos[..., 2] < self.mrobot_cfg.motion.foot_contact_height_threshold).float()
        self.ref_euler_xyz = _quat_wxyz_to_euler_xyz(self.ref_waist_quat[:, 0, :])
        self.normalized_bpm_cmd = (self.bpm_cmd - self.reference_bpm_mean) / self.reference_bpm_std
        self._refresh_tracking_ref_buffers()

    def _compute_trajectory_ref_state(self):
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
        if getattr(self, "_use_feet_pos_z_for_ref_foot_height", False):
            self.ref_foot_height[:] = self.ref_feet_pos[..., 2]
        else:
            self.ref_foot_height[:] = self.foot_height_buffer[self.ref_idx, phase_ids]
        self.ref_root_linvel[:] = self.root_linvel_buffer[self.ref_idx, phase_ids]
        self.ref_root_angvel[:] = self.root_angvel_buffer[self.ref_idx, phase_ids]
        self.ref_euler_xyz[:] = self.euler_xyz_buffer[self.ref_idx, phase_ids]
        self.normalized_bpm_cmd.zero_()
        self._refresh_tracking_ref_buffers()

    def _refresh_tracking_ref_buffers(self):
        self._fill_tracking_ref_buffers(
            self.tracking_ref_pos_buf,
            self.tracking_ref_quat_buf,
            self.tracking_ref_lin_vel_buf,
            self.tracking_ref_ang_vel_buf,
            self._tracking_ref_specs,
        )
        self._fill_tracking_ref_buffers(
            self.priv_tracking_ref_pos_buf,
            self.priv_tracking_ref_quat_buf,
            self.priv_tracking_ref_lin_vel_buf,
            self.priv_tracking_ref_ang_vel_buf,
            self._priv_tracking_ref_specs,
        )
        self._tracking_cache_valid = False

    def _ref_tensors_for_kind(self, kind):
        if kind == "pelvis":
            return self.ref_pelvis_pos, self.ref_pelvis_quat, self.ref_pelvis_vel, self.ref_pelvis_ang_vel
        if kind == "feet":
            return self.ref_feet_pos, self.ref_feet_quat, self.ref_feet_vel, self.ref_feet_ang_vel
        if kind == "knee":
            return self.ref_knee_pos, self.ref_knee_quat, self.ref_knee_vel, self.ref_knee_ang_vel
        if kind == "hip":
            return self.ref_hip_pos, self.ref_hip_quat, self.ref_hip_vel, self.ref_hip_ang_vel
        if kind == "pelvic_yaw":
            return self.ref_pelvic_yaw_pos, self.ref_pelvic_yaw_quat, self.ref_pelvic_yaw_vel, self.ref_pelvic_yaw_ang_vel
        if kind == "waist":
            return self.ref_waist_pos, self.ref_waist_quat, self.ref_waist_vel, self.ref_waist_ang_vel
        raise RuntimeError(f"Unknown tracking reference kind: {kind}")

    def _fill_tracking_ref_buffers(self, pos_buf, quat_buf, lin_vel_buf, ang_vel_buf, specs):
        for out_idx, (kind, part_idx) in enumerate(specs):
            ref_pos, ref_quat, ref_vel, ref_ang_vel = self._ref_tensors_for_kind(kind)
            pos_buf[:, out_idx] = ref_pos[:, part_idx]
            quat_buf[:, out_idx] = ref_quat[:, part_idx]
            lin_vel_buf[:, out_idx] = ref_vel[:, part_idx]
            ang_vel_buf[:, out_idx] = ref_ang_vel[:, part_idx]

    def _advance_reference_phase(self):
        if self._uses_trajectory_reference():
            max_phase = self.demo_lengths[self.ref_idx].clamp(min=1) - 1
            self.phase_idx[:] = torch.minimum(self.phase_idx + 1, max_phase)
            return
        self.phase_rad[:] = torch.remainder(
            self.phase_rad + (2.0 * math.pi * self.bpm_cmd / 60.0) * self.step_dt,
            2.0 * math.pi,
        )

    def _get_actor_reference_extra_obs(self):
        if self._uses_trajectory_reference():
            return torch.zeros(self.num_envs, 0, device=self.device)
        return torch.cat((torch.sin(self.phase_rad), torch.cos(self.phase_rad), self.normalized_bpm_cmd), dim=-1)

    def _get_privileged_reference_phase_obs(self):
        if self._uses_trajectory_reference():
            return torch.zeros_like(self.phase_rad)
        return self.phase_rad / (2.0 * math.pi)

    def _init_adaptive_phase_sampling(self):
        motion = self.mrobot_cfg.motion
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
        if not self._uses_trajectory_reference():
            return
        motion = self.mrobot_cfg.motion
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
        if not self._uses_trajectory_reference():
            return
        motion = self.mrobot_cfg.motion
        if not bool(getattr(motion, "use_adaptive_phase_sampling", False)):
            return
        if env_ids is None or len(env_ids) == 0:
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

    def _sample_trajectory_phase_starts(self, ref_ids):
        phase = self._sample_uniform_trajectory_phase_starts(ref_ids)
        if len(ref_ids) == 0:
            return phase
        motion = self.mrobot_cfg.motion
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

    def _pred_column(self, pred, name, default):
        idx = self.reference_column_index.get(name)
        if idx is None:
            return torch.full((self.num_envs,), float(default), device=self.device)
        return pred[:, idx]

    def _get_current_anchor_pose(self):
        return self.rigid_state[:, self.waist_body_id, :3], self.rigid_state[:, self.waist_body_id, 3:7]

    def _get_current_anchor_pose_local(self):
        anchor_pos_w, anchor_quat = self._get_current_anchor_pose()
        return anchor_pos_w - self.terrain.env_origins, anchor_quat

    def _get_anchor_yaw_alignment(self, cur_anchor_quat=None):
        if cur_anchor_quat is None:
            _, cur_anchor_quat = self._get_current_anchor_pose()
        ref_anchor_quat = self.ref_waist_quat[:, 0, :]
        q_diff = _quat_mul_wxyz(cur_anchor_quat, _quat_inv_wxyz(ref_anchor_quat))
        return _calc_heading_quat_wxyz(q_diff)

    def _align_ref_positions_to_current_anchor(self, ref_body_pos, cur_anchor_pos=None, cur_anchor_quat=None):
        if cur_anchor_pos is None or cur_anchor_quat is None:
            cur_anchor_pos, cur_anchor_quat = self._get_current_anchor_pose_local()
        ref_anchor_pos = self.ref_waist_pos[:, 0, :]
        yaw_diff_quat = self._get_anchor_yaw_alignment(cur_anchor_quat)
        rel_ref_pos = ref_body_pos - ref_anchor_pos.unsqueeze(1)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_pos.shape[1], -1)
        rotated_rel_ref = _quat_apply_wxyz(yaw_diff_repeat.reshape(-1, 4), rel_ref_pos.reshape(-1, 3)).reshape_as(rel_ref_pos)
        target_pos = cur_anchor_pos.unsqueeze(1) + rotated_rel_ref
        target_pos[:, :, 2] = ref_body_pos[:, :, 2]
        return target_pos

    def _align_ref_quats_to_current_anchor(self, ref_body_quat, cur_anchor_quat=None):
        yaw_diff_quat = self._get_anchor_yaw_alignment(cur_anchor_quat)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_quat.shape[1], -1)
        return _quat_mul_wxyz(yaw_diff_repeat.reshape(-1, 4), ref_body_quat.reshape(-1, 4)).reshape_as(ref_body_quat)

    def _align_ref_vectors_to_current_anchor(self, ref_body_vec, cur_anchor_quat=None):
        yaw_diff_quat = self._get_anchor_yaw_alignment(cur_anchor_quat)
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_body_vec.shape[1], -1)
        return _quat_apply_wxyz(yaw_diff_repeat.reshape(-1, 4), ref_body_vec.reshape(-1, 3)).reshape_as(ref_body_vec)

    def get_ref_rel_state_current(self, ref_pos_w, ref_quat_w):
        r_pos_w = self.ref_waist_pos[:, 0, :]
        r_quat_w = self.ref_waist_quat[:, 0, :]
        r_inv_quat = _quat_inv_wxyz(r_quat_w)
        num_parts = ref_pos_w.shape[1]
        diff_p = ref_pos_w - r_pos_w.unsqueeze(1)
        r_inv_repeat = r_inv_quat.unsqueeze(1).expand(-1, num_parts, -1).reshape(-1, 4)
        rel_p = _quat_apply_wxyz(r_inv_repeat, diff_p.reshape(-1, 3)).reshape(self.num_envs, num_parts, 3)
        rel_q = _quat_mul_wxyz(r_inv_repeat, ref_quat_w.reshape(-1, 4)).reshape(self.num_envs, num_parts, 4)
        return rel_p.reshape(self.num_envs, -1), rel_q.reshape(self.num_envs, -1)

    def get_rel_pose(self, indices, root_pos, root_quat):
        pos_w = self.rigid_state[:, indices, :3]
        quat_w = self.rigid_state[:, indices, 3:7]
        num_parts = indices.shape[0]
        p_rel_w = pos_w - root_pos.unsqueeze(1)
        root_repeat = root_quat.unsqueeze(1).expand(-1, num_parts, -1).reshape(-1, 4)
        p_rel_b = _quat_rotate_inverse_wxyz(root_repeat, p_rel_w.reshape(-1, 3)).reshape(self.num_envs, num_parts * 3)
        inv_root_repeat = _quat_inv_wxyz(root_quat).unsqueeze(1).expand(-1, num_parts, -1).reshape(-1, 4)
        q_rel_b = _quat_mul_wxyz(inv_root_repeat, quat_w.reshape(-1, 4)).reshape(self.num_envs, num_parts * 4)
        return p_rel_b, q_rel_b

    def _get_aligned_body_pos_targets(self, indices, ref_body_pos):
        cur_body_pos = self.rigid_state[:, indices, :3] - self.terrain.env_origins.unsqueeze(1)
        cur_anchor_pos, cur_anchor_quat = self._get_current_anchor_pose_local()
        target_pos = self._align_ref_positions_to_current_anchor(ref_body_pos, cur_anchor_pos, cur_anchor_quat)
        return cur_body_pos, target_pos

    def _get_aligned_body_quat_targets(self, indices, ref_body_quat):
        cur_body_quat = self.rigid_state[:, indices, 3:7]
        _, cur_anchor_quat = self._get_current_anchor_pose()
        target_quat = self._align_ref_quats_to_current_anchor(ref_body_quat, cur_anchor_quat)
        return cur_body_quat, target_quat

    def _get_aligned_body_vector_targets(self, indices, ref_body_vec, state_slice):
        cur_body_vec = self.rigid_state[:, indices, state_slice]
        _, cur_anchor_quat = self._get_current_anchor_pose()
        target_vec = self._align_ref_vectors_to_current_anchor(ref_body_vec, cur_anchor_quat)
        return cur_body_vec, target_vec

    def _tracking_ref_pos(self):
        return self.tracking_ref_pos_buf

    def _tracking_ref_quat(self):
        return self.tracking_ref_quat_buf

    def _tracking_ref_lin_vel(self):
        return self.tracking_ref_lin_vel_buf

    def _tracking_ref_ang_vel(self):
        return self.tracking_ref_ang_vel_buf

    def _get_tracking_reward_cache(self):
        if self._tracking_cache_valid and self._tracking_cache_common_step == self.common_step_counter:
            return self._tracking_reward_cache
        cur_anchor_pos, cur_anchor_quat = self._get_current_anchor_pose_local()
        yaw_diff_quat = self._get_anchor_yaw_alignment(cur_anchor_quat)
        ref_pos = self._tracking_ref_pos()
        ref_quat = self._tracking_ref_quat()
        ref_lin_vel = self._tracking_ref_lin_vel()
        ref_ang_vel = self._tracking_ref_ang_vel()
        yaw_diff_repeat = yaw_diff_quat.unsqueeze(1).expand(-1, ref_pos.shape[1], -1)

        rel_ref_pos = ref_pos - self.ref_waist_pos[:, 0, :].unsqueeze(1)
        target_pos = cur_anchor_pos.unsqueeze(1) + _quat_apply_wxyz(
            yaw_diff_repeat.reshape(-1, 4),
            rel_ref_pos.reshape(-1, 3),
        ).reshape_as(rel_ref_pos)
        target_pos[:, :, 2] = ref_pos[:, :, 2]
        target_quat = _quat_mul_wxyz(yaw_diff_repeat.reshape(-1, 4), ref_quat.reshape(-1, 4)).reshape_as(ref_quat)
        target_lin_vel = _quat_apply_wxyz(yaw_diff_repeat.reshape(-1, 4), ref_lin_vel.reshape(-1, 3)).reshape_as(ref_lin_vel)
        target_ang_vel = _quat_apply_wxyz(yaw_diff_repeat.reshape(-1, 4), ref_ang_vel.reshape(-1, 3)).reshape_as(ref_ang_vel)

        self._tracking_reward_cache = (
            self.rigid_state[:, self.all_tracking_indices, :3] - self.terrain.env_origins.unsqueeze(1),
            target_pos,
            self.rigid_state[:, self.all_tracking_indices, 3:7],
            target_quat,
            self.rigid_state[:, self.all_tracking_indices, 7:10],
            target_lin_vel,
            self.rigid_state[:, self.all_tracking_indices, 10:13],
            target_ang_vel,
        )
        self._tracking_cache_common_step = self.common_step_counter
        self._tracking_cache_valid = True
        return self._tracking_reward_cache

    def debug_check_keypoint_alignment(self, env_ids=None):
        """Print a one-shot keypoint alignment summary for manual debugging."""
        if env_ids is None:
            env_ids = torch.arange(min(self.num_envs, 4), device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            cur_body_pos, target_pos, cur_body_quat, target_quat, cur_body_vel, target_vel, cur_body_ang_vel, target_ang_vel = (
                self._get_tracking_reward_cache()
            )
            pos_err = torch.norm(cur_body_pos[env_ids] - target_pos[env_ids], dim=-1)
            quat_err = _quat_error_mag_wxyz(
                cur_body_quat[env_ids].reshape(-1, 4),
                target_quat[env_ids].reshape(-1, 4),
            ).reshape(len(env_ids), -1)
            lin_vel_err = torch.norm(cur_body_vel[env_ids] - target_vel[env_ids], dim=-1)
            ang_vel_err = torch.norm(cur_body_ang_vel[env_ids] - target_ang_vel[env_ids], dim=-1)
            print(
                "[MRobot keypoint debug] body_order="
                + ", ".join(self._body_names_from_indices(self.all_tracking_indices)),
                flush=True,
            )
            print(
                "[MRobot keypoint debug] ref_order follows tracking_body_names",
                flush=True,
            )
            print(
                "[MRobot keypoint debug] ref_specs="
                + ", ".join([f"{kind}[{part_idx}]" for kind, part_idx in self._tracking_ref_specs]),
                flush=True,
            )
            print(
                "[MRobot keypoint debug] "
                f"pos_err_mean={pos_err.mean().item():.6f}, pos_err_max={pos_err.max().item():.6f}, "
                f"quat_err_mean={quat_err.mean().item():.6f}, quat_err_max={quat_err.max().item():.6f}, "
                f"lin_vel_err_mean={lin_vel_err.mean().item():.6f}, "
                f"ang_vel_err_mean={ang_vel_err.mean().item():.6f}",
                flush=True,
            )

    def _quat_err_6d(self, q_curr_flat, q_ref_flat, num_parts):
        q_c = q_curr_flat.reshape(-1, 4)
        q_r = q_ref_flat.reshape(-1, 4)
        err_q = _quat_mul_wxyz(_quat_conjugate_wxyz(q_c), q_r)
        err_mat = _matrix_from_quat_wxyz(err_q)
        err_6d = err_mat[..., :2].reshape(-1, 6)
        return err_6d.reshape(self.num_envs, num_parts * 6)

    def _update_state_cache(self, force=False):
        if not force and self._state_cache_valid and self._state_cache_common_step == self.common_step_counter:
            return
        total_profile_start = self._profile_section_start()
        profile_start = self._profile_section_start()
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        self.base_quat = self.robot.data.root_quat_w
        self.base_euler_xyz = _quat_wxyz_to_euler_xyz(self.base_quat)
        self._profile_section_end("state_root_joint", profile_start)
        profile_start = self._profile_section_start()
        self.base_lin_vel = _quat_rotate_inverse_wxyz(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = _quat_rotate_inverse_wxyz(self.base_quat, self.root_states[:, 10:13])
        self._profile_section_end("state_base_vel", profile_start)
        profile_start = self._profile_section_start()
        self.rigid_state = self.robot.data.body_state_w
        self._profile_section_end("state_body", profile_start)
        self._state_cache_common_step = self.common_step_counter
        self._state_cache_valid = True
        self._tracking_cache_valid = False
        self._profile_section_end("state_cache", total_profile_start)

    def _pre_physics_step(self, actions):
        actions = torch.clip(actions, -self.mrobot_cfg.normalization.clip_actions, self.mrobot_cfg.normalization.clip_actions)
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.actions[:] = actions
        # Reference state is computed for the current policy time in
        # _get_observations()/reset.  Recomputing here doubles reference-net
        # forwards without changing the target used by this action.
        self.full_actions.zero_()
        self.full_actions[:, self.num_control] = actions
        profile_start = self._profile_section_start()
        self.full_actions[:, self.num_notcontrol] = self.ref_dof_pos[:, self.ref_num_notcontrol] / self.cfg.action_scale
        self._profile_section_end("noncontrolled_ref_action", profile_start)
        self._apply_substep = 0

    def _apply_action(self):
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        if self.obs_imu_delay_buffer is not None:
            self._record_sys_delay_state()
        profile_start = self._profile_section_start()
        if self.mrobot_cfg.normalization.actions_filter:
            rate = float(self._apply_substep + 1) / float(max(self.cfg.decimation, 1))
            full_actions = (1.0 - rate) * self.last_full_actions + rate * self.full_actions
        else:
            full_actions = self.full_actions
        scaled = full_actions * self.cfg.action_scale
        self._profile_section_end("action_filter", profile_start)
        profile_start = self._profile_section_start()
        if self.action_delay_buffer is not None and getattr(self.mrobot_cfg.domain_rand, "action_delay", False):
            self.action_delay_buffer[:, :, self.action_delay_write_idx] = scaled
            read_idx = torch.remainder(
                self.action_delay_write_idx - self.action_delay_timestep.long(),
                self.action_delay_buffer_size,
            )
            scaled = self.action_delay_buffer[self.env_ids_arange, :, read_idx]
            self.action_delay_write_idx = (self.action_delay_write_idx + 1) % self.action_delay_buffer_size
        self._profile_section_end("action_delay", profile_start)
        self.delayed_full_actions_scaled[:] = scaled
        target = self.target_dof_pos
        target.copy_(self.ref_dof_pos)
        if getattr(self.mrobot_cfg.control, "use_ref_residual_target", False):
            target[:, self.num_control] = self.ref_dof_pos[:, self.num_control] + scaled[:, self.num_control]
        else:
            default_dof_pos_with_offset = self.default_dof_pos + self.default_dof_pos_offsets
            target[:, self.num_control] = default_dof_pos_with_offset[:, self.num_control] + scaled[:, self.num_control]
        target.add_(self.motor_offsets)
        dof_vel_for_pd = self._get_ankle_dq_for_pd()
        torques = self.Kp_factors * self.p_gains * (target - self.dof_pos) - self.Kd_factors * self.d_gains * dof_vel_for_pd
        torques = torques * self.motor_strength_factors
        self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        if getattr(self.mrobot_cfg.domain_rand, "use_coulomb", False):
            left = (
                self.mrobot_cfg.domain_rand.left_Us
                * torch.tanh(self.dof_vel[:, [4, 5]] / self.mrobot_cfg.domain_rand.left_Qs)
                + self.mrobot_cfg.domain_rand.left_Ud * self.dof_vel[:, [4, 5]]
            )
            right = (
                self.mrobot_cfg.domain_rand.right_Us
                * torch.tanh(self.dof_vel[:, [10, 11]] / self.mrobot_cfg.domain_rand.right_Qs)
                + self.mrobot_cfg.domain_rand.right_Ud * self.dof_vel[:, [10, 11]]
            )
            self.torques[:, [4, 5]] -= left
            self.torques[:, [10, 11]] -= right
            self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
        self.sim_order_torques.zero_()
        self.sim_order_torques[:, self.joint_sim_ids] = self.torques
        self.backend.set_dof_targets(self.sim_order_torques)
        self._apply_substep += 1

    def _get_observations(self):
        if self._uses_trajectory_reference():
            # IsaacGym Dance uses LeggedRobot.post_physics_step: reward is
            # computed against the current trajectory frame, then the reference
            # phase is advanced for the next policy observation/action.  DirectRLEnv
            # calls _get_dones() before _get_rewards(), so the trajectory branch
            # advances here instead of in _get_dones().
            profile_start = self._profile_section_start()
            self._advance_reference_phase()
            self.compute_ref_state()
            self._profile_section_end("reference_update", profile_start)
        self._update_state_cache()
        anchor_pos_w, anchor_quat_w = self._get_current_anchor_pose()
        anchor_pos_local, _ = self._get_current_anchor_pose_local()
        tracking_p_b, tracking_q_b = self.get_rel_pose(self.all_tracking_indices, anchor_pos_w, anchor_quat_w)
        ref_p0, ref_q0 = self.get_ref_rel_state_current(self._tracking_ref_pos(), self._tracking_ref_quat())
        num_tracking_parts = len(self.all_tracking_indices)
        ref_anchor_pos = self.ref_waist_pos[:, 0, :]
        ref_anchor_quat = self.ref_waist_quat[:, 0, :]
        anchor_pos_b = _quat_rotate_inverse_wxyz(anchor_quat_w, ref_anchor_pos - anchor_pos_local)
        anchor_quat_b = _quat_mul_wxyz(_quat_conjugate_wxyz(anchor_quat_w), ref_anchor_quat)
        anchor_ori_b = _matrix_from_quat_wxyz(anchor_quat_b)[..., :2].reshape(self.num_envs, -1)
        tracking_err_p = tracking_p_b - ref_p0
        tracking_err_q = self._quat_err_6d(tracking_q_b, ref_q0, num_tracking_parts)
        if getattr(self.mrobot_cfg.domain_rand, "sys_delay", False) and self.obs_imu_delay_buffer is not None:
            root_states_obs, dof_pos_vel_obs = self._get_sys_delayed_obs_state()
            q = (dof_pos_vel_obs[:, : self.mrobot_cfg.env.num_actions] - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = dof_pos_vel_obs[:, self.mrobot_cfg.env.num_actions :] * self.obs_scales.dof_vel
            base_ang_vel_obs = _quat_rotate_inverse_wxyz(root_states_obs[:, 3:7], root_states_obs[:, 10:13])
            base_euler_obs = _quat_wxyz_to_euler_xyz(root_states_obs[:, 3:7])
        else:
            q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = self.dof_vel * self.obs_scales.dof_vel
            base_ang_vel_obs = self.base_ang_vel
            base_euler_obs = self.base_euler_xyz[:, 0:3]
        q, dq = self._apply_actor_ankle_obs_bias(q, dq)
        obs_euler = base_euler_obs.clone()
        obs_euler[:, 2] = _wrap_to_pi(obs_euler[:, 2] - self.initial_base_yaw)
        obs_now = torch.cat(
            (
                q[:, self.num_control],
                dq[:, self.num_control],
                self.actions,
                base_ang_vel_obs * self.obs_scales.ang_vel,
                obs_euler * self.obs_scales.quat,
                self._get_actor_reference_extra_obs(),
            ),
            dim=-1,
        )
        ref_waist_euler = _quat_wxyz_to_euler_xyz(self.ref_waist_quat[:, 0, :])
        goal_terms = [
            self.ref_dof_pos[:, self.num_control] * self.obs_scales.dof_pos,
            self.ref_dof_vel[:, self.num_control] * self.obs_scales.dof_vel,
            self.ref_waist_pos[:, 0, 2:3],
            ref_waist_euler[:, 0:2],
            self.ref_waist_vel[:, 0, :] * self.obs_scales.lin_vel,
            self.ref_waist_ang_vel[:, 0, 2:3] * self.obs_scales.ang_vel,
        ]
        if int(getattr(self.mrobot_cfg.env, "num_goal_obs", 31)) >= 33:
            goal_terms.append(self.ref_feet_contact.float())
        goal_buf = torch.cat(goal_terms, dim=-1)
        if self.mrobot_cfg.noise.add_noise:
            obs_now = obs_now + (2.0 * torch.rand_like(obs_now) - 1.0) * self.noise_scale_vec * self.mrobot_cfg.noise.noise_level
        self.policy_obs_buf[:, : self.mrobot_cfg.env.num_single_obs] = obs_now
        self.policy_obs_buf[:, self.mrobot_cfg.env.num_single_obs :] = goal_buf
        default_dof_pos_with_offset = self.default_dof_pos + self.default_dof_pos_offsets
        priv_hist = torch.cat(
            (
                self.root_states[:, 2:3],
                self.base_euler_xyz[:, 0:2],
                (self.dof_pos - default_dof_pos_with_offset)[:, self.num_control] * self.obs_scales.dof_pos,
                dq[:, self.num_control],
                self.actions,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
            ),
            dim=-1,
        )
        priv_curr = self.priv_curr_buf
        priv_curr.zero_()
        priv_curr[:, 0:3] = anchor_pos_b
        priv_curr[:, 3:9] = anchor_ori_b
        priv_curr[:, 9:12] = (self.base_lin_vel - self.ref_root_linvel) * self.obs_scales.lin_vel
        priv_curr[:, 12:15] = (self.base_ang_vel - self.ref_root_angvel) * self.obs_scales.ang_vel
        cursor = 15
        tracking_pos_dim = num_tracking_parts * 3
        tracking_quat_dim = num_tracking_parts * 6
        priv_curr[:, cursor : cursor + tracking_pos_dim] = tracking_err_p
        cursor += tracking_pos_dim
        priv_curr[:, cursor : cursor + tracking_quat_dim] = tracking_err_q
        cursor += tracking_quat_dim
        dr = self.mrobot_cfg.domain_rand
        priv_curr[:, cursor : cursor + 2] = self.rand_push_force[:, :2] / max(float(getattr(dr, "max_push_vel_xy", 1.0)), 1e-6)
        cursor += 2
        priv_curr[:, cursor : cursor + 3] = self.rand_push_torque / max(float(getattr(dr, "max_push_ang_vel", 1.0)), 1e-6)
        cursor += 3
        priv_curr[:, cursor : cursor + 3] = self.disturbance_force[:, 0, :] / max(
            abs(float(getattr(dr, "disturbance_range", [-1.0, 1.0])[1])), 1e-6
        )
        cursor += 3
        friction_range = getattr(dr, "static_friction_range", getattr(dr, "friction_range", [0.0, 1.0]))
        restitution_range = getattr(dr, "restitution_range", [0.0, 1.0])
        priv_curr[:, cursor : cursor + 1] = (self.friction_coeffs - friction_range[0]) / max(
            float(friction_range[1] - friction_range[0]), 1e-6
        )
        cursor += 1
        priv_curr[:, cursor : cursor + 1] = (self.restitution_coeffs - restitution_range[0]) / max(
            float(restitution_range[1] - restitution_range[0]), 1e-6
        )
        cursor += 1
        priv_curr[:, cursor : cursor + 12] = (self.Kp_factors[:, self.num_control] - dr.kp_range[0]) / max(
            float(dr.kp_range[1] - dr.kp_range[0]), 1e-6
        )
        cursor += 12
        priv_curr[:, cursor : cursor + 12] = (self.Kd_factors[:, self.num_control] - dr.kd_range[0]) / max(
            float(dr.kd_range[1] - dr.kd_range[0]), 1e-6
        )
        cursor += 12
        payload_range = getattr(dr, "payload_mass_range", [0.0, 1.0])
        priv_curr[:, cursor : cursor + 1] = (self.payload - payload_range[0]) / max(float(payload_range[1] - payload_range[0]), 1e-6)
        cursor += 1
        priv_curr[:, cursor : cursor + 3] = self.com_displacement * self.obs_scales.com_pos
        cursor += 3
        priv_curr[:, cursor : cursor + 2] = 1.0 - self.ref_feet_contact
        cursor += 2
        priv_curr[:, cursor : cursor + 1] = self._get_privileged_reference_phase_obs()
        cursor += 1
        if cursor != self.priv_curr_dim:
            raise RuntimeError(f"MRobot IsaacLab privileged current dim mismatch: filled={cursor}, cfg={self.priv_curr_dim}")
        self.privileged_obs_buf[:, :45] = priv_hist
        self.privileged_obs_buf[:, 45 : 45 + self.priv_curr_dim] = priv_curr
        self.privileged_obs_buf[:, 45 + self.priv_curr_dim :] = goal_buf
        if not getattr(self, "_printed_observation_layout_shapes", False):
            print(
                "[HumanoidGym-Ex] Observation layout changed: Dance actor obs 61/73 -> 75 "
                "or BPM actor obs 64 -> 76. "
                "Old checkpoints and normalizer statistics are incompatible. "
                "Train from scratch or reset normalizer.",
                flush=True,
            )
            print(
                "[HumanoidGym-Ex] MRobot IsaacLab observation shapes: "
                f"obs_now.shape={tuple(obs_now.shape)}, goal_buf.shape={tuple(goal_buf.shape)}, "
                f"actor obs.shape={tuple(self.policy_obs_buf.shape)}, "
                f"privileged obs.shape={tuple(self.privileged_obs_buf.shape)}",
                flush=True,
            )
            self._printed_observation_layout_shapes = True
        if self.policy_obs_buf.shape[1] != self.mrobot_cfg.env.num_observations:
            raise RuntimeError(
                f"MRobot IsaacLab obs dim mismatch: got {self.policy_obs_buf.shape[1]}, "
                f"cfg={self.mrobot_cfg.env.num_observations}"
            )
        if self.privileged_obs_buf.shape[1] != self.mrobot_cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"MRobot IsaacLab privileged obs dim mismatch: got {self.privileged_obs_buf.shape[1]}, "
                f"cfg={self.mrobot_cfg.env.num_privileged_obs}"
            )
        clip_obs = float(getattr(self.mrobot_cfg.normalization, "clip_observations", 50.0))
        self.policy_obs_buf.clamp_(min=-clip_obs, max=clip_obs)
        self.privileged_obs_buf.clamp_(min=-clip_obs, max=clip_obs)
        return {"policy": self.policy_obs_buf, "critic": self.privileged_obs_buf}

    def _prepare_reward_function(self):
        self.reward_scales = {}
        for name in dir(self.mrobot_cfg.rewards.scales):
            if name.startswith("_"):
                continue
            value = getattr(self.mrobot_cfg.rewards.scales, name)
            if callable(value) or value == 0:
                continue
            self.reward_scales[name] = value * self.step_dt
        self.reward_names = [name for name in self.reward_scales if name != "termination"]
        missing_rewards = [name for name in self.reward_names if not hasattr(self, "_reward_" + name)]
        if missing_rewards:
            raise AttributeError(
                "MRobot IsaacLab reward functions missing for enabled scales: "
                + ", ".join("_reward_" + name for name in missing_rewards)
            )
        self.reward_functions = [getattr(self, "_reward_" + name) for name in self.reward_names]
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_scales}
        self.tracking_score_names = [
            name
            for name in self.reward_names
            if name.startswith("imitation") or name.startswith("imition") or name == "teleop_contact_mask"
        ]
        self.tracking_score_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.tracking_score_names
        }

    def _write_common_episode_infos(self, episode_info, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            zero = torch.zeros((), device=self.device)
            episode_info["mean_episode_length"] = zero
            episode_info["fall_ratio"] = zero
            episode_info["fall_contact_ratio"] = zero
            episode_info["base_too_low_ratio"] = zero
            episode_info["ref_end_ratio"] = zero
            episode_info["time_out_ratio"] = zero
            episode_info["tracking_error_ratio"] = zero
            episode_info["waist_z_bad_ratio"] = zero
            episode_info["waist_ori_bad_ratio"] = zero
            episode_info["foot_z_bad_ratio"] = zero
        else:
            episode_info["mean_episode_length"] = torch.mean(self.curriculum_episode_length_buf[env_ids].float())
            episode_info["fall_ratio"] = torch.mean(self.fall_reset_buf[env_ids].float())
            episode_info["fall_contact_ratio"] = torch.mean(self.contact_reset_buf[env_ids].float())
            episode_info["base_too_low_ratio"] = torch.mean(self.base_too_low_buf[env_ids].float())
            episode_info["ref_end_ratio"] = torch.mean(self.ref_end_reset_buf[env_ids].float())
            episode_info["time_out_ratio"] = torch.mean(self.time_out_buf[env_ids].float())
            episode_info["tracking_error_ratio"] = torch.mean(self.tracking_error_reset_buf[env_ids].float())
            episode_info["waist_z_bad_ratio"] = torch.mean(self.waist_z_bad_buf[env_ids].float())
            episode_info["waist_ori_bad_ratio"] = torch.mean(self.waist_ori_bad_buf[env_ids].float())
            episode_info["foot_z_bad_ratio"] = torch.mean(self.foot_z_bad_buf[env_ids].float())
        if self._uses_trajectory_reference():
            episode_info["sampling_entropy"] = self.motion_sampling_entropy
            episode_info["sampling_top1_prob"] = self.motion_sampling_top1_prob
            episode_info["sampling_top1_bin"] = self.motion_sampling_top1_bin
        episode_info["curriculum_stage"] = torch.tensor(float(max(self._domain_rand_curriculum_stage, 0)), device=self.device)
        episode_info["push_ratio"] = torch.tensor(float(self._current_push_ratio), device=self.device)
        episode_info["disturbance_ratio"] = torch.tensor(float(self._current_disturbance_ratio), device=self.device)
        episode_info["restitution_ratio"] = torch.tensor(float(self._current_restitution_ratio), device=self.device)
        episode_info["pd_ratio"] = torch.tensor(float(self._current_pd_ratio), device=self.device)
        episode_info["motor_strength_ratio"] = torch.tensor(float(self._current_motor_strength_ratio), device=self.device)
        episode_info["delay_ratio"] = torch.tensor(float(self._current_delay_ratio), device=self.device)

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)
        self.extras.pop("episode", None)
        for name, func in zip(self.reward_names, self.reward_functions):
            raw_rew = func()
            rew = raw_rew * self.reward_scales[name]
            rewards += rew
            self.episode_sums[name] += rew
            if name in self.tracking_score_sums:
                self.tracking_score_sums[name] += raw_rew
        if "termination" in self.reward_scales:
            # Match Gym mimic: terminal reward only penalizes real falls, not
            # normal timeouts or reference-trajectory end resets.
            rew = self.fall_reset_buf.float() * self.reward_scales["termination"]
            rewards += rew
            self.episode_sums["termination"] += rew
        self.last_full_actions[:] = self.full_actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques
        return rewards

    def _post_physics_step_callback(self):
        dr = self.mrobot_cfg.domain_rand
        if getattr(dr, "push_robots", False):
            if self.common_step_counter >= self.next_push_step:
                self._push_robots()
                self.next_push_step = self.common_step_counter + self._sample_push_interval_steps()
            self._clear_external_forces()
        elif getattr(dr, "disturbance", False) and self.common_step_counter % self.disturbance_interval == 0:
            self._disturbance_robots()
        else:
            self._clear_external_forces()

    def _push_robots(self):
        dr = self.mrobot_cfg.domain_rand
        max_vel = float(getattr(dr, "max_push_vel_xy", 0.0))
        max_ang = float(getattr(dr, "max_push_ang_vel", 0.0))
        self.rand_push_force[:, :2] = _torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), self.device)
        self.rand_push_force[:, 2] = 0.0
        self.rand_push_torque[:] = _torch_rand_float(-max_ang, max_ang, (self.num_envs, 3), self.device)
        root_velocity = self.robot.data.root_vel_w.clone()
        root_velocity[:, 0:2] += self.rand_push_force[:, :2]
        root_velocity[:, 3:6] = self.rand_push_torque
        self.robot.write_root_velocity_to_sim(root_velocity)

    def _disturbance_robots(self):
        dr = self.mrobot_cfg.domain_rand
        disturbance = _torch_rand_float(dr.disturbance_range[0], dr.disturbance_range[1], (self.num_envs, 1, 3), self.device)
        self.disturbance_force[:] = disturbance
        self.disturbance_torque.zero_()
        body_ids = torch.tensor([self.disturbance_body_id], dtype=torch.long, device=self.device)
        self.robot.set_external_force_and_torque(
            forces=self.disturbance_force,
            torques=self.disturbance_torque,
            body_ids=body_ids,
            env_ids=torch.arange(self.num_envs, device=self.device),
            is_global=False,
        )
        self.external_force_active[:] = True
        self._external_force_active_any = True

    def _clear_external_forces(self, env_ids=None):
        if not self._external_force_active_any:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        forces = torch.zeros(len(env_ids), 1, 3, device=self.device)
        body_ids = torch.tensor([self.disturbance_body_id], dtype=torch.long, device=self.device)
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=False,
        )
        self.external_force_active[env_ids] = False
        if len(env_ids) == self.num_envs:
            self._external_force_active_any = False

    def _get_dones(self):
        profile_start = self._profile_section_start()
        if self._uses_trajectory_reference():
            # Keep Dance reward on the current phase_idx; _get_observations()
            # advances to the next trajectory frame after reward/reset.
            pass
        else:
            self._advance_reference_phase()
            self.compute_ref_state()
        self._profile_section_end("dones_phase_ref", profile_start)
        profile_start = self._profile_section_start()
        self._update_state_cache()
        self._profile_section_end("dones_state_cache", profile_start)
        profile_start = self._profile_section_start()
        self.curriculum_episode_length_buf += 1
        self._post_physics_step_callback()
        self._profile_section_end("dones_push_contact", profile_start)
        profile_start = self._profile_section_start()
        self.contact_forces = self.backend.get_contact_forces()
        self._profile_section_end("dones_contact_read", profile_start)
        profile_start = self._profile_section_start()
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        contact_died = torch.zeros_like(self.time_out_buf)
        if len(self.termination_contact_indices) > 0:
            contact_died = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        grace_mask = self.episode_length_buf > 5
        self.contact_reset_buf = contact_died & grace_mask
        self.base_too_low_buf = (self.root_states[:, 2] < 0.5) & grace_mask
        if self._uses_trajectory_reference():
            max_phase = self.demo_lengths[self.ref_idx].clamp(min=1) - 1
            self.ref_end_reset_buf = self.phase_idx >= max_phase
            term_cfg = getattr(self.mrobot_cfg, "termination", None)
            if term_cfg is not None and bool(getattr(term_cfg, "use_tracking_error_termination", False)):
                tracking_grace = self.episode_length_buf > int(getattr(term_cfg, "tracking_termination_grace_steps", 5))
                ref_waist_z = self.ref_waist_pos[:, 0, 2]
                cur_waist_z = self.rigid_state[:, self.waist_body_id, 2]
                self.waist_z_bad_buf = (
                    torch.abs(ref_waist_z - cur_waist_z) > float(getattr(term_cfg, "waist_z_threshold", 0.25))
                ) & tracking_grace
                ref_projected_gravity = _quat_rotate_inverse_wxyz(self.ref_waist_quat[:, 0, :], self.gravity_vec)
                cur_projected_gravity = _quat_rotate_inverse_wxyz(
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
        else:
            self.ref_end_reset_buf[:] = False
            self.waist_z_bad_buf[:] = False
            self.waist_ori_bad_buf[:] = False
            self.foot_z_bad_buf[:] = False
            self.tracking_error_reset_buf[:] = False
        self.fall_reset_buf = self.contact_reset_buf | self.base_too_low_buf | self.tracking_error_reset_buf
        self.adaptive_phase_failure_buf = (
            self.fall_reset_buf & (~self.time_out_buf) & (~self.ref_end_reset_buf)
        )
        # DirectRLEnv ORs reset_terminated with reset_time_outs after this return.
        self.reset_buf = self.fall_reset_buf | self.ref_end_reset_buf
        self._profile_section_end("dones_termination", profile_start)
        return self.reset_buf, self.time_out_buf

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        profile_start = self._profile_section_start()
        if hasattr(self, "episode_sums"):
            self.extras["episode"] = {}
            denom = self.episode_length_buf[env_ids].float() + 1.0
            episode_lengths = self.curriculum_episode_length_buf[env_ids].float()
            dr = self.mrobot_cfg.domain_rand
            adaptive_min_iteration = getattr(dr, "adaptive_min_iteration", 0)
            if self._adaptive_curriculum_current_iteration >= adaptive_min_iteration and len(env_ids) > 0:
                self._adaptive_curriculum_length_sum += torch.sum(episode_lengths)
                self._adaptive_curriculum_fall_sum += torch.sum(self.fall_reset_buf[env_ids].float())
                self._adaptive_curriculum_pending_resets += int(len(env_ids))
            for name in self.episode_sums:
                self.extras["episode"]["rew_" + name] = torch.mean(self.episode_sums[name][env_ids] / denom)
                self.episode_sums[name][env_ids] = 0.0
            if hasattr(self, "tracking_score_sums"):
                for name in self.tracking_score_sums:
                    self.extras["episode"]["score_" + name] = torch.mean(
                        self.tracking_score_sums[name][env_ids] / denom
                    )
                    self.tracking_score_sums[name][env_ids] = 0.0
            self._write_common_episode_infos(self.extras["episode"], env_ids)
        self._profile_section_end("reset_episode_logging", profile_start)
        profile_start = self._profile_section_start()
        if self._uses_trajectory_reference():
            self._update_adaptive_phase_failures(env_ids)
        self._profile_section_end("reset_adaptive_sampling", profile_start)
        profile_start = self._profile_section_start()
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._profile_section_end("reset_robot_super", profile_start)
        profile_start = self._profile_section_start()
        motion = self.mrobot_cfg.motion
        if self._uses_trajectory_reference():
            if getattr(self.mrobot_cfg.domain_rand, "RSI", 1):
                self.ref_idx[env_ids] = torch.randint(0, self.data_length, (len(env_ids),), device=self.device)
                self.phase_idx[env_ids] = self._sample_trajectory_phase_starts(self.ref_idx[env_ids])
            else:
                self.ref_idx[env_ids] = 0
                self.phase_idx[env_ids] = 0
            self.phase_rad[env_ids] = 0.0
            self.normalized_bpm_cmd[env_ids] = 0.0
        else:
            if motion.fixed_bpm is not None:
                bpm = torch.full((len(env_ids), 1), float(motion.fixed_bpm), device=self.device)
            elif motion.sample_integer_bpm:
                bpm_min = int(round(float(motion.bpm_range[0])))
                bpm_max = int(round(float(motion.bpm_range[1])))
                if bpm_max < bpm_min:
                    raise ValueError(f"Invalid bpm_range: {motion.bpm_range}")
                num_regular_bpms = bpm_max - bpm_min + 1
                if motion.include_zero_bpm:
                    bpm_choice = torch.randint(0, num_regular_bpms + 1, (len(env_ids), 1), device=self.device)
                    bpm = torch.where(
                        bpm_choice == 0,
                        torch.zeros_like(bpm_choice),
                        bpm_choice + bpm_min - 1,
                    ).float()
                else:
                    bpm = torch.randint(bpm_min, bpm_max + 1, (len(env_ids), 1), device=self.device).float()
            else:
                bpm = _torch_rand_float(float(motion.bpm_range[0]), float(motion.bpm_range[1]), (len(env_ids), 1), self.device)
                if motion.include_zero_bpm:
                    zero_mask = torch.rand(len(env_ids), 1, device=self.device) < 0.5
                    bpm = torch.where(zero_mask, torch.zeros_like(bpm), bpm)
            self.bpm_cmd[env_ids] = bpm
            if getattr(motion, "randomize_init_phase", True):
                self.init_phase_rad[env_ids] = _torch_rand_float(
                    float(motion.init_phase_range[0]), float(motion.init_phase_range[1]), (len(env_ids), 1), self.device
                )
            else:
                self.init_phase_rad[env_ids] = 0.0
            self.phase_rad[env_ids] = self.init_phase_rad[env_ids]
        self.compute_ref_state()
        self._profile_section_end("reset_bpm_ref", profile_start)
        profile_start = self._profile_section_start()
        self._randomize_reset_buffers(env_ids)
        self._profile_section_end("reset_domain_rand", profile_start)
        profile_start = self._profile_section_start()
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.terrain.env_origins[env_ids]
        root[:, 7:13] = 0.0
        if self._uses_trajectory_reference():
            ref_root = self.root_states_buffer[self.ref_idx[env_ids], self.phase_idx[env_ids]]
            root[:, :3] = self.terrain.env_origins[env_ids] + ref_root[:, :3]
            root[:, 3:7] = _quat_xyzw_to_wxyz(ref_root[:, 3:7])
            root[:, 7:10] = self.root_linvel_buffer[self.ref_idx[env_ids], self.phase_idx[env_ids]]
            root[:, 10:13] = self.root_angvel_buffer[self.ref_idx[env_ids], self.phase_idx[env_ids]]
        else:
            root[:, 7:9] = _torch_rand_float(-0.1, 0.1, (len(env_ids), 2), self.device)
        dr = self.mrobot_cfg.domain_rand
        if not self.cfg.deterministic_reset and getattr(dr, "randomize_root_xy_reset", False):
            xy_range = getattr(dr, "root_xy_reset_range", [0.0, 0.0])
            root[:, 0:2] += _torch_rand_float(float(xy_range[0]), float(xy_range[1]), (len(env_ids), 2), self.device)
        if not self.cfg.deterministic_reset and getattr(dr, "randomize_root_yaw_reset", False):
            yaw_range = getattr(dr, "root_yaw_reset_range", [0.0, 0.0])
            yaw_noise = _torch_rand_float(float(yaw_range[0]), float(yaw_range[1]), (len(env_ids),), self.device)
            root[:, 3:7] = _quat_mul_wxyz(_quat_from_yaw_wxyz(yaw_noise), root[:, 3:7])
        self.initial_base_yaw[env_ids] = _quat_wxyz_to_euler_xyz(root[:, 3:7])[:, 2]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        ref_joint_pos = self.ref_dof_pos[env_ids].clone()
        if not self.cfg.deterministic_reset and getattr(dr, "randomize_init_dof_pos", False):
            init_range = getattr(dr, "init_dof_pos_range", [-0.03, 0.03])
            ref_joint_pos += _torch_rand_float(
                float(init_range[0]), float(init_range[1]), (len(env_ids), len(self.joint_sim_ids)), self.device
            )
        lower = self.dof_pos_limits[env_ids, :, 0]
        upper = self.dof_pos_limits[env_ids, :, 1]
        ref_joint_pos = torch.minimum(torch.maximum(ref_joint_pos, lower), upper)
        joint_pos[:, self.joint_sim_ids] = ref_joint_pos
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel[:, self.joint_sim_ids] = self.ref_dof_vel[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._profile_section_end("reset_state_write", profile_start)
        profile_start = self._profile_section_start()
        self.dof_pos = joint_pos[:, self.joint_sim_ids]
        self.dof_vel = joint_vel[:, self.joint_sim_ids]
        self._prime_sys_delay_state(env_ids, root_state=root, dof_pos=ref_joint_pos, dof_vel=self.ref_dof_vel[env_ids])
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.full_actions[env_ids] = 0.0
        self.last_full_actions[env_ids] = 0.0
        self.delayed_full_actions_scaled[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.fall_reset_buf[env_ids] = False
        self.contact_reset_buf[env_ids] = False
        self.base_too_low_buf[env_ids] = False
        self.ref_end_reset_buf[env_ids] = False
        self.tracking_error_reset_buf[env_ids] = False
        self.waist_z_bad_buf[env_ids] = False
        self.waist_ori_bad_buf[env_ids] = False
        self.foot_z_bad_buf[env_ids] = False
        self.adaptive_phase_failure_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        self.reset_buf[env_ids] = False
        self.curriculum_episode_length_buf[env_ids] = 0
        self._clear_external_forces(env_ids)
        self._update_state_cache(force=True)
        self._profile_section_end("reset_cleanup", profile_start)

    def _reward_imitation_whole_body_pos(self):
        cur_body_pos, target_pos, _, _, _, _, _, _ = self._get_tracking_reward_cache()
        dist_sq = torch.sum(torch.square(cur_body_pos - target_pos), dim=-1).mean(dim=1)
        return torch.exp(-dist_sq / (self.mrobot_cfg.rewards.sigma.whole_body_pos ** 2))

    def _reward_imitation_whole_body_rot(self):
        _, _, cur_body_quat, target_quat, _, _, _, _ = self._get_tracking_reward_cache()
        rot_error = _quat_error_mag_wxyz(cur_body_quat.reshape(-1, 4), target_quat.reshape(-1, 4)).reshape(self.num_envs, -1)
        return torch.exp(-torch.square(rot_error).mean(dim=-1) / (self.mrobot_cfg.rewards.sigma.whole_body_rot ** 2))

    def _reward_imitation_whole_body_lin_vel(self):
        _, _, _, _, cur_body_vel, target_vel, _, _ = self._get_tracking_reward_cache()
        error = torch.sum(torch.square(cur_body_vel - target_vel), dim=-1).mean(dim=-1)
        return torch.exp(-error / (self.mrobot_cfg.rewards.sigma.whole_body_lin_vel ** 2))

    def _reward_imitation_whole_body_ang_vel(self):
        _, _, _, _, _, _, cur_body_ang_vel, target_ang_vel = self._get_tracking_reward_cache()
        error = torch.sum(torch.square(cur_body_ang_vel - target_ang_vel), dim=-1).mean(dim=-1)
        return torch.exp(-error / (self.mrobot_cfg.rewards.sigma.whole_body_ang_vel ** 2))

    def _reward_imition_root_pos(self):
        cur_root_pos, _ = self._get_current_anchor_pose_local()
        diff_sq = torch.sum(torch.square(cur_root_pos - self.ref_waist_pos[:, 0, :]), dim=-1)
        return torch.exp(-diff_sq / (self.mrobot_cfg.rewards.sigma.root_pos ** 2))

    def _reward_imition_root_rot(self):
        _, cur_q = self._get_current_anchor_pose()
        return torch.exp(-_quat_error_mag_wxyz(cur_q, self.ref_waist_quat[:, 0, :]) ** 2 / (self.mrobot_cfg.rewards.sigma.root_rot ** 2))

    def _reward_imition_joint_pos(self):
        pos_target = self.ref_dof_pos[:, self.num_control]
        joint_pos = self.dof_pos[:, self.num_control]
        diff = joint_pos - pos_target
        err = torch.sum(self.dof_err_w * torch.square(diff), dim=1)
        sigma_dof_pos = 0.5
        return torch.exp(-err / (sigma_dof_pos ** 2))

    def _reward_imition_joint_vel(self):
        vel_target = self.ref_dof_vel[:, self.num_control].clamp(-10.0, 10.0)
        joint_vel = self.dof_vel[:, self.num_control]
        diff = vel_target - joint_vel
        err = torch.sum(self.dof_err_w * torch.square(diff), dim=1)
        sigma_dof_vel = 2.0
        return torch.exp(-err / (sigma_dof_vel ** 2))

    def _reward_imition_foot_height(self):
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        diff = feet_z - self.ref_foot_height
        return torch.exp(-torch.square(torch.norm(diff, dim=1)) / (self.mrobot_cfg.rewards.sigma.foot_height ** 2))

    def _reward_imition_root_height(self):
        cur_waist_z = self.rigid_state[:, self.waist_body_id, 2]
        ref_waist_z = self.ref_waist_pos[:, 0, 2]
        return torch.exp(-torch.square(cur_waist_z - ref_waist_z) / (self.mrobot_cfg.rewards.sigma.root_height ** 2))

    def _reward_imitation_root_vel(self):
        cur_root_vel = self.rigid_state[:, self.waist_body_id, 7:10]
        ref_root_vel = self.ref_waist_vel[:, 0, :]
        diff_sq = torch.sum(torch.square(cur_root_vel - ref_root_vel), dim=-1)
        return torch.exp(-diff_sq / (self.mrobot_cfg.rewards.sigma.root_vel ** 2))

    def _reward_imition_base_ang_vel(self):
        cur_ang_vel = self.rigid_state[:, self.waist_body_id, 10:13]
        ref_ang_vel = self.ref_waist_ang_vel[:, 0, :]
        diff_sq = torch.sum(torch.square(cur_ang_vel - ref_ang_vel), dim=-1)
        return torch.exp(-diff_sq / (self.mrobot_cfg.rewards.sigma.root_ang_vel ** 2))

    def _reward_teleop_contact_mask(self):
        if len(self.feet_contact_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        contact_force_z = self.contact_forces[:, self.feet_contact_indices, 2]
        cur_contact_mask = (contact_force_z > 20.0).float()
        ref_contact_mask = self.ref_feet_contact.float()
        error_contact_mask = torch.abs(cur_contact_mask - ref_contact_mask)
        reward = 1.0 - error_contact_mask.mean(dim=-1)
        ref_lift = 1.0 - ref_contact_mask
        drag_force = (contact_force_z - 20.0).clamp(min=0.0) * ref_lift
        return reward - 0.01 * drag_force.mean(dim=-1)

    def _reward_foot_slip(self):
        if len(self.feet_contact_indices) != 2 or len(self.feet_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_contact_indices, 2] > 20.0
        foot_speed_xy = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        return torch.sum(torch.sqrt(foot_speed_xy.clamp(min=0.0)) * contact.float(), dim=1)

    def _reward_pre_landing_foot_z_vel(self):
        if len(self.feet_contact_indices) != 2 or len(self.feet_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        ref_swing = 1.0 - self.ref_feet_contact.float()
        contact_force_z = self.contact_forces[:, self.feet_contact_indices, 2]
        no_contact = (contact_force_z < 5.0).float()
        foot_z = self.rigid_state[:, self.feet_indices, 2]
        near_ground = ((foot_z > 0.055) & (foot_z < 0.12)).float()
        foot_vel_z = self.rigid_state[:, self.feet_indices, 9]
        downward_vel_excess = torch.clamp(-foot_vel_z - 0.15, min=0.0)
        pre_landing_mask = ref_swing * no_contact * near_ground
        return torch.sum(torch.square(downward_vel_excess) * pre_landing_mask, dim=1)

    def _reward_feet_contact_forces(self):
        if len(self.feet_contact_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)
        max_contact_force = float(getattr(self.mrobot_cfg.rewards, "max_contact_force", 600.0))
        foot_contact_force = torch.norm(self.contact_forces[:, self.feet_contact_indices, :], dim=-1)
        return torch.sum((foot_contact_force - max_contact_force).clip(min=0.0, max=5000.0), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, self.num_control]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.num_control]), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel[:, self.num_control] - self.dof_vel[:, self.num_control]) / self.step_dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_ankle_dof_acc(self):
        ankle_idx = self.ankle_reward_indices
        return torch.sum(torch.square((self.last_dof_vel[:, ankle_idx] - self.dof_vel[:, ankle_idx]) / self.step_dt), dim=1)

    def _reward_ankle_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.ankle_reward_indices]), dim=1)

    def _reward_dof_pos_limits(self):
        dof_pos = self.dof_pos[:, self.num_control]
        dof_limits = self.dof_pos_limits[:, self.num_control, :]
        out_of_limits = -(dof_pos - dof_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (dof_pos - dof_limits[:, :, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        soft_limit_val = self.torque_limits * 0.9
        torques_to_check = torch.abs(self.torques[:, self.num_control])
        relevant_soft_limits = soft_limit_val[self.num_control]
        over_limit = torques_to_check - relevant_soft_limits
        violation = torch.clamp(over_limit, min=0.0)
        reward = torch.mean(violation / (self.torque_limits[self.num_control] * 0.1).clamp(min=1e-6), dim=1)
        return torch.clamp(reward, min=0.0, max=1.0)

    def _reward_ankle_torque_limit(self):
        ankle_idx = self.ankle_reward_indices
        soft_limit_val = self.torque_limits[ankle_idx] * 0.9
        violation = torch.clamp(torch.abs(self.torques[:, ankle_idx]) - soft_limit_val, min=0.0)
        reward = torch.mean(violation / (self.torque_limits[ankle_idx] * 0.1).clamp(min=1e-6), dim=1)
        return torch.clamp(reward, min=0.0, max=1.0)


class MrobotMimicDanceIsaacLabEnv(MrobotMimicIsaacLabEnv):
    """IsaacLab dance env entry point.

    The runtime implementation is intentionally shared with
    MrobotMimicIsaacLabEnv.  Switching ``mrobot_cfg_cls`` to
    MrobotMimicDanceLabCfg makes the parent use the trajectory-reference
    branches for observation, reset, reference phase update, reward config and
    policy/action spaces.
    """

    cfg: MrobotMimicDanceIsaacLabEnvCfg
    mrobot_cfg_cls = MrobotMimicDanceLabCfg
