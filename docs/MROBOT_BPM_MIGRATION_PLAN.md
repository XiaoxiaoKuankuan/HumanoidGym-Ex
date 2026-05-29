# MRobot BPM/Music Dance 迁移说明

本文档记录把旧仓库 `/home/weil/hl_rl` 中非 AMP 的 MRobot BPM/music dance mimic 训练逻辑迁移到当前仓库 `/home/weil/HumanoidGym-Ex` 的实现结果、文件清单、代码接口、运行流程和验证方式。

## 1. 迁移目标

本次迁移的目标是让当前框架同时支持：

1. IsaacGym 版本的 `mrobot_music` mimic 训练任务，尽量对齐旧仓库训练行为。
2. IsaacLab/IsaacSim DirectRLEnv 版本的 MRobot BPM mimic 训练入口，用于 IsaacSim 流程。
3. BPM reference state network 的训练/播放脚本，但不携带旧数据集或模型。
4. `sim2sim_mimic.py` 部署脚本，保留 BPM 控制、reference model、ONNX/JIT policy、Space4Bar 和诊断绘图能力。

明确不迁移的内容：

1. 不迁移 AMP runner、AMP storage、discriminator、replay buffer、AMP env。
2. 不迁移旧数据集、旧 reference checkpoint、旧 policy checkpoint、日志、deploy model。
3. 不迁移除 `sim2sim_mimic.py` 之外的其他 sim2sim 脚本。
4. 不迁移旧的数据生成脚本，后续如果需要可以单独做。

## 2. 核心接口约定

`mrobot_music` 的训练接口保持旧逻辑的核心 shape：

| 项目 | 数值 | 说明 |
| --- | ---: | --- |
| actor observation | 64 | 45 维 proprio + 19 维 goal |
| privileged observation | 210 | 45 维历史/本体 + 146 维当前特权项 + 19 维 goal |
| policy action | 12 | 策略只输出双腿 12 个受控关节 residual |
| full action / full DOF | 29 | 环境内部仍维护 29 DOF，全身 PD target |
| aux target | 9 | 可选 aux supervision，当前 MRobot mimic 启用 |
| task name | `mrobot_music` | 在 task registry 中注册 |

BPM reference 逻辑保持旧逻辑：

1. reset 时采样 `bpm_cmd`。
2. 支持 `cfg.motion.fixed_bpm` 固定 BPM。
3. 支持 `cfg.motion.include_zero_bpm` 采样静止 BPM。
4. 支持 `cfg.motion.sample_integer_bpm` 整数 BPM 采样。
5. 支持 `cfg.motion.randomize_init_phase` 随机初始相位。
6. 相位推进为 `phase_rad += 2*pi*bpm/60*dt`，然后 wrap 到 `[0, 2*pi)`。
7. reference checkpoint 不存在时显式抛出 `FileNotFoundError`，不会从旧仓库隐式复制模型。

## 3. 旧仓库到新仓库的主要映射

| 旧仓库位置 | 新仓库位置 | 说明 |
| --- | --- | --- |
| `/home/weil/hl_rl/hl_rl/humanoid/envs/custom/mrobot_mimic_config.py` | `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config.py` | MRobot BPM mimic 配置 |
| `/home/weil/hl_rl/hl_rl/humanoid/envs/custom/mrobot_mimic_env.py` | `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_env.py` | IsaacGym mimic env |
| `/home/weil/hl_rl/hl_rl/humanoid/envs/base/base_task.py` | `humanoid_gym_ex/envs/robots/mrobot/mrobot_base_task.py` | MRobot 专用 BaseTask，避免改动现有 XBot base |
| `/home/weil/hl_rl/hl_rl/humanoid/envs/base/legged_robot.py` | `humanoid_gym_ex/envs/robots/mrobot/mrobot_legged_robot.py` | MRobot 专用 LeggedRobot base |
| 旧 PPO optional normalizer / aux 逻辑 | `humanoid_gym_ex/algo/ppo/*` | 以可选能力接入，默认不影响现有任务 |
| 旧 BPM reference 脚本 | `humanoid_gym_ex/scripts/bpm/*` | 只迁移脚本，不迁移数据/模型 |
| 旧 `sim2sim_mimic.py` | `humanoid_gym_ex/scripts/sim2sim_mimic.py` | 只迁移这个 sim2sim 脚本 |
| 旧 CASBOT/MRobot asset | `resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/` 和 `resources/robots/Mrobot/` | 只复制 robot asset，不复制数据和 checkpoint |

## 4. 新增文件说明

### `docs/MROBOT_BPM_MIGRATION_PLAN.md`

当前说明文档。记录迁移范围、文件功能、运行命令、测试方式和注意事项。

### `humanoid_gym_ex/envs/robots/mrobot/__init__.py`

MRobot env 包入口。导出：

1. `MrobotMimicCfg`
2. `MrobotMimicCfgPPO`
3. `MrobotMimicEnv`

如果当前 Python 会话没有可用 IsaacGym，它允许配置类仍可被导入，便于脚本和静态工具读取配置。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_base_task.py`

MRobot 专用 IsaacGym `BaseTask`。来自旧仓库 base task，用来支撑 MRobot mimic env 的旧式 IsaacGym tensor API 流程。

保留为 MRobot 私有 base 的原因：

1. 避免直接改动当前仓库已有 XBot task 的 base 类。
2. 兼容旧 MRobot reset、step、viewer、buffer 初始化方式。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_legged_robot.py`

MRobot 专用 IsaacGym `LeggedRobot` base。核心功能：

1. 维护 29 维 full action buffer。
2. 接收 12 维 policy action，并写入 `self.actions[:, self.num_control]`。
3. 非受控关节通过 `_get_noncontrolled_ref_actions()` 跟随 reference。
4. 创建 IsaacGym sim、terrain、asset、env。
5. 初始化 DOF/body/contact/reward/domain randomization buffer。
6. 支持 `self.num_policy_actions = len(self.num_control)`，PPO 可以只按 12 维动作建 policy。
7. `is_amp=False`，AMP 分支不启用，也没有迁移 AMP runner/storage/discriminator。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config.py`

MRobot BPM mimic 基础配置。核心内容：

1. `env.num_observations = 64`。
2. `env.num_privileged_obs = 210`。
3. `env.num_actions = 29`。
4. `env.num_policy_actions = 12`。
5. `env.num_aux = 9`。
6. `env.num_control = [0..11]`，策略只控制双腿。
7. `env.num_notcontrol = [12..28]`，腰、上肢、头跟随 reference。
8. `motion.reference_model_path` 默认为 `deploy/reference_state_keypoint_model.pt`。
9. `motion.bpm_range = [60.0, 170.0]`，支持 zero BPM、整数 BPM、随机 phase。
10. 保留旧 PD、domain randomization、reward scale、normalizer、PPO 配置。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_gym.py`

IsaacGym 训练配置入口。它继承 `mrobot_mimic_config.py`，保持旧 IsaacGym 训练行为不变，并把 `runner.save_config` 指向 `mrobot_mimic_config_gym.py`。`mrobot_music` task registry 默认使用这个配置入口。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py`

IsaacLab/IsaacSim 训练配置入口。它继承基础配置，并加入 IsaacLab 专用差异：

1. `domain_rand.static_friction_range` 和 `domain_rand.dynamic_friction_range` 分开配置。
2. `lab_joint_effort_limits`、`lab_joint_velocity_limits`、`lab_joint_position_limits` 显式列出从 URDF 解析出的 29 个 canonical 关节 limit。
3. IsaacLab actuator 的 `effort_limit_sim` 和 `velocity_limit_sim` 使用上述显式 limit，避免 importer/actuator 默认值不透明。
4. `randomize_joint_armature=False` 时仍会把旧配置中的 `joint_armature_values` 写入 IsaacLab PhysX，和 IsaacGym `_process_dof_props()` 行为对齐。
5. `runner.experiment_name` 默认为 `mrobot_mimic_May_music_BPM_isaaclab`。
6. `runner.save_config` 指向 `mrobot_mimic_config_lab.py`。

### `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_env.py`

IsaacGym MRobot BPM mimic env。核心功能：

1. 加载 BPM reference checkpoint。
2. 根据 `bpm_cmd + phase_rad` 预测全身 reference state。
3. 根据 joint alias 将 reference 输出列映射到当前 robot DOF。
4. 构造 64 维 actor obs。
5. 构造 210 维 privileged obs。
6. 构造 9 维 aux target。
7. 策略输出为 12 维腿部 residual action。
8. 非控制关节跟随 reference。
9. 控制关节默认使用 `ref_dof_pos + residual` 作为 PD target。
10. 迁移旧非 AMP mimic reward：whole body pos/rot/vel、root pos/rot、dof acc、action rate、ankle penalty、limit penalty 等。
11. reset 时采样 BPM、phase，并按 reference 初始化 DOF/root。

### `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`

IsaacLab/IsaacSim DirectRLEnv 版本的 MRobot BPM mimic env。核心功能：

1. 使用 IsaacLab `DirectRLEnv`。
2. 使用 `MrobotMimicLabCfg` 的 robot、PD、obs/action/reward 语义。
3. 使用相同 `ReferenceStateNet` 和 BPM/phase 编码。
4. `action_space=12`，`observation_space=64`，`state_space=210`。
5. 使用当前仓库已有的 `IsaacLabBackend` 做 IsaacLab tensor 适配。
6. plane terrain 默认使用本地生成的静态薄 cuboid，不依赖 IsaacLab 远程 `default_environment.usd`，便于无网络或受限网络环境 smoke。
7. 只在 IsaacLab 初始姿态中将 `upper_left_7_joint` / `upper_right_7_joint` 夹到 URDF limit 内，避免 IsaacLab articulation 初始化阶段直接报错；共享 MRobot IsaacGym 配置不改。
8. reset 时先采样 BPM/phase，再计算 reference state，并把 29 个 canonical DOF 初始化到 reference 姿态，避免从默认站姿被 PD 猛拉到舞蹈参考姿态。
9. 每个 physics substep 都重新读取 IsaacLab 最新 joint pos/vel 后计算 PD torque，避免 decimation 内用旧状态重复算力矩。
10. torque limit 使用 URDF/IsaacLab joint effort limit，避免上肢/头部被统一 250Nm 过大限幅驱动。
11. 实现 IsaacLab 可写入的 domain randomization：摩擦/恢复系数、质量、COM、joint armature/friction、PD/motor 系数、动作延迟、push 和外力扰动。

注意：IsaacLab 版本是 IsaacSim 训练入口，目标是语义对齐和可训练；与 IsaacGym 旧环境的低层 PhysX 接触细节不会 100% 数值等价。

### `humanoid_gym_ex/utils/reference_state.py`

BPM reference 共享模块。包含：

1. `JOINT_NAME_ALIASES`：旧数据列名到新 URDF DOF 名的 alias。
2. `ReferenceStateNet`：BPM/phase 到 reference state 的 MLP。
3. `encode_bpm_phase()`：输入编码为 `[normalized_bpm, sin(phase), cos(phase)]`。

IsaacGym env、IsaacLab env、sim2sim 脚本共同使用这个语义。

### `humanoid_gym_ex/utils/torch_utils.py`

从旧仓库迁移的 quaternion/heading 工具函数。MRobot mimic reward 和 body alignment 逻辑会用到，例如：

1. `calc_heading_quat()`
2. `calc_heading_quat_inv()`
3. quaternion 到 exp-map / tangent-normal 的转换工具

### `humanoid_gym_ex/algo/ppo/normalizer.py`

旧仓库中的 running mean/std observation normalizer。功能：

1. `EmpiricalNormalization.update()` 更新统计量。
2. `EmpiricalNormalization.act()` 用冻结统计量归一化。
3. checkpoint 保存/加载时可选保存 normalizer 状态。

MRobot config 中 `env.normalize_obs=True` 时启用；其他任务默认不受影响。

### `humanoid_gym_ex/scripts/bpm/__init__.py`

BPM reference 脚本包入口。

### `humanoid_gym_ex/scripts/bpm/train_reference_state_network.py`

BPM reference state network 训练脚本。输入是外部 BPM keypoint CSV 数据目录，输出 `.pt` checkpoint，可选导出 ONNX。

该脚本不包含旧数据路径硬编码，运行时需要显式指定：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py \
  --data-dir /path/to/bpm_keypoint_csv_dir \
  --output /path/to/reference_state_keypoint_model.pt
```

### `humanoid_gym_ex/scripts/bpm/play_reference_state_network.py`

BPM reference state network 播放/检查脚本。用于给定 checkpoint 后按 BPM/phase 生成 reference 轨迹并可视化/检查输出。

### `humanoid_gym_ex/scripts/sim2sim_mimic.py`

迁移的唯一 sim2sim 脚本。保留：

1. JIT `.pt` policy 加载。
2. ONNX policy 加载。
3. reference model 加载。
4. 64 维 policy input 构造。
5. 12 维 action output 使用。
6. BPM 键盘控制。
7. Space4Bar 地形。
8. MuJoCo viewer。
9. 诊断绘图。

`mujoco`、`mujoco_viewer`、`glfw`、`pygame`、`onnxruntime` 被改成可选导入；缺依赖时会在真正运行对应功能时给出清晰错误，`--help` 可以正常打开。

### `deploy/export_actor.py`

用户导入的 policy 导出脚本，本次已适配当前仓库。功能：

1. 从当前 `MrobotMimicCfg` / `MrobotMimicCfgPPO` 读取 obs/action/aux/hidden/std 配置，避免继续硬编码旧仓库路径和旧维度。
2. 支持从训练 checkpoint `model_*.pt` 直接导出 ONNX。
3. 如果 checkpoint 中包含 `obs_normalizer`，导出时会把观测归一化层一起 baked 到 ONNX 中，部署端输入 raw obs。
4. 兼容可选 JIT policy 导出路径。
5. 使用当前仓库的 `humanoid_gym_ex/algo/ppo/actor_critic.py` 重建 actor。

推荐训练完成后使用：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python deploy/export_actor.py \
  --ckpt_path logs/mrobot_mimic_May_music_BPM/<run_dir>/model_<iter>.pt \
  -o deploy/casbot_mrobot_bpm.onnx
```

### `humanoid_gym_ex/scripts/space4bar.py`

`sim2sim_mimic.py` 需要的 Space4Bar 地形/障碍辅助代码。

### `humanoid_gym_ex/scripts/train_mrobot_isaaclab.py`

IsaacLab/IsaacSim 训练入口。功能：

1. 启动 IsaacLab `AppLauncher`。
2. 构造 `MrobotMimicIsaacLabEnvCfg`。
3. 设置 reference checkpoint、num envs、iteration、steps per env。
4. 用当前仓库 PPO `OnPolicyRunner` 训练。
5. 默认 `WANDB_MODE=offline`。

### `humanoid_gym_ex/scripts/play_mrobot_isaaclab.py`

IsaacLab/IsaacSim 播放入口。可加载 TorchScript policy；不提供 policy 时可用 zero action 做 env smoke。

### `resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/`

MRobot mimic 默认使用的 CASBOT02 ENCOS 7dof shell 机器人 asset，包含 URDF 和 mesh。默认 URDF：

```text
resources/robots/CASBOT02_ENCOS_7dof_shell_20251015/Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015_bass.urdf
```

### `resources/robots/Mrobot/`

`sim2sim_mimic.py` 使用的 MRobot/MuJoCo 相关 asset。只迁移 robot/terrain asset，不包含训练数据或模型。

## 5. 修改文件说明

### `humanoid_gym_ex/envs/__init__.py`

修改内容：

1. 注册原有 `humanoid_ppo`。
2. 新增注册 `mrobot_music`。
3. 新增 `register_tasks()` 可重试注册函数。
4. 如果当前会话无法加载 IsaacGym，允许配置导入继续进行；训练/播放入口会在真正运行时重试注册。

这样做的原因是 IsaacGym 的 `gymtorch` extension 在某些环境下会因为 cache 目录不可写而在静态导入时失败。配置文件和脚本 help 不应该因此完全不可用。

### `humanoid_gym_ex/envs/robots/humanoid_env.py`

修改内容：

1. 将 `LeggedRobot` 改为从 `humanoid_gym_ex.envs.base.legged_robot` 直接导入。

原因：避免 `humanoid_gym_ex.envs` 包级注册时触发循环导入，保证 XBot 和 MRobot task 都可以注册。

### `humanoid_gym_ex/algo/ppo/actor_critic.py`

修改内容：

1. `ActorCritic` 新增 `num_aux=0`。
2. 当 `num_aux>0` 时创建 aux head。
3. 新增 `get_aux()`。
4. `init_noise_std` 支持 scalar、list、numpy array，MRobot 使用 12 维 action std。
5. 支持 `fixed_std`。

默认 `num_aux=0` 时，现有任务仍然是原 actor/critic 行为。

### `humanoid_gym_ex/algo/ppo/rollout_storage.py`

修改内容：

1. `Transition` 新增 `aux` 字段。
2. `RolloutStorage` 新增可选 `aux_shape`。
3. `aux_shape=None` 或 `0` 时不分配 aux buffer。
4. mini-batch generator 可选返回 `aux_batch`。

这样 MRobot mimic 能使用旧仓库 aux supervision，其他任务不需要改 action/obs/storage。

### `humanoid_gym_ex/algo/ppo/ppo.py`

修改内容：

1. 新增 `aux_loss_coef=0.01`。
2. `init_storage()` 支持可选 `aux_shape`。
3. `process_env_step()` 支持从参数或 `infos["aux"]` 接收 aux target。
4. `update()` 中当存在 aux target 和 aux head 时，计算 MSE aux loss。
5. 返回 `(mean_value_loss, mean_surrogate_loss, mean_aux_loss)`。

默认无 aux 时 `mean_aux_loss=0.0`，普通 PPO 行为保持不变。

### `humanoid_gym_ex/algo/ppo/on_policy_runner.py`

修改内容：

1. 支持 `env.num_policy_actions`，没有时回退到 `env.num_actions`。
2. 支持 `env.num_aux` 和 `cfg.env.num_aux`。
3. storage 使用 12 维 policy action，而不是 MRobot 的 29 维 full action。
4. 支持 env.step 返回 6 元组：`obs, privileged_obs, rewards, dones, infos, aux`。
5. 支持 observation normalizer。
6. checkpoint 保存/加载 optional normalizer。
7. 日志新增 `Loss/aux_loss`。
8. 如果 env 有 `update_domain_rand_curriculum()`，runner 会调用它。

### `humanoid_gym_ex/algo/ppo/__init__.py`

修改内容：

1. 导出 `EmpiricalNormalization`。

### `humanoid_gym_ex/utils/__init__.py`

修改内容：

1. 暴露 `ReferenceStateNet`、`JOINT_NAME_ALIASES`、`encode_bpm_phase`。
2. 对 IsaacGym import-order/cache 问题做保护，允许 reference/config-only 脚本导入。

### `humanoid_gym_ex/utils/helpers.py`

修改内容：

1. `get_args()` 新增 `--reference_model`。
2. `get_args()` 新增 `--fixed_bpm`。
3. `update_cfg_from_args()` 支持覆盖 `cfg.motion.reference_model_path`。
4. `update_cfg_from_args()` 支持覆盖 `cfg.motion.fixed_bpm`。

这样 IsaacGym 训练/播放不需要改 Python 配置文件也能指定外部 reference checkpoint。

### `humanoid_gym_ex/scripts/train.py`

修改内容：

1. 直接运行脚本时自动把仓库根目录加入 `sys.path`。
2. 真正训练前调用 `envs_module.register_tasks()`，确保 `mrobot_music` 已注册。

### `humanoid_gym_ex/scripts/play.py`

修改内容：

1. 直接运行脚本时自动把仓库根目录加入 `sys.path`。
2. 真正播放前调用 `envs_module.register_tasks()`。
3. 兼容 MRobot env.step 返回 6 元组，其中第 6 项是 aux。

## 5.5 为什么 `robots/mrobot` 比 `robots/xbot` 文件多

`humanoid_gym_ex/envs/robots/xbot/` 目录当前主要只有 IsaacLab 相关适配文件：

| XBot 文件 | 功能 |
| --- | --- |
| `humanoid_gym_ex/envs/robots/xbot/isaaclab_env.py` | XBot 的 IsaacLab DirectRLEnv 实现 |
| `humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py` | 把 IsaacLab env 包装成当前 PPO runner 能使用的 VecEnv 接口 |

XBot 的 IsaacGym 训练核心并不放在 `robots/xbot/` 目录下，而是复用了当前仓库原本已有的公共文件，例如：

| XBot 复用文件 | 功能 |
| --- | --- |
| `humanoid_gym_ex/envs/robots/humanoid_config.py` | XBot 原始配置 |
| `humanoid_gym_ex/envs/robots/humanoid_env.py` | XBot 原始 IsaacGym 任务 |
| `humanoid_gym_ex/envs/base/legged_robot.py` | 当前仓库原有 LeggedRobot base |
| `humanoid_gym_ex/envs/base/base_task.py` | 当前仓库原有 BaseTask |

MRobot 目录文件更多，是因为这次迁移的是旧仓库 `/home/weil/hl_rl` 中一整套非 AMP BPM mimic 任务。虽然整体框架相似，但 MRobot 的动作、观测、reward、reference network、12 维 policy action 到 29 维 full action 的映射、非受控关节 reference 跟随，都和 XBot 原始任务不同。为了不破坏当前 XBot 任务，本次没有直接改公共 `base_task.py` / `legged_robot.py`，而是在 `robots/mrobot/` 下放了一套 MRobot 私有 IsaacGym 任务文件。

MRobot 文件职责如下：

| MRobot 文件 | 后端 | 功能 |
| --- | --- | --- |
| `humanoid_gym_ex/envs/robots/mrobot/__init__.py` | 通用 | MRobot 包入口，导出 `MrobotMimicCfg`、`MrobotMimicCfgPPO`、`MrobotMimicEnv`；同时对 IsaacGym 导入失败做保护，便于配置/脚本 help 可用 |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config.py` | 通用 | MRobot BPM mimic 的核心配置：robot asset、29 DOF、12 policy action、64 obs、210 privileged obs、BPM/reference、PD、reward、domain randomization、PPO 默认参数 |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_gym.py` | IsaacGym | IsaacGym 训练配置入口，继承基础配置，`mrobot_music` task registry 默认使用它 |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py` | IsaacLab/IsaacSim | IsaacLab 训练配置入口，继承基础配置，单独放置 IsaacLab 摩擦范围、armature 写入语义、实验名和 `save_config` |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_base_task.py` | IsaacGym | 从旧仓库迁移的 MRobot 私有 BaseTask，负责 IsaacGym sim/viewer/buffer 的基础生命周期 |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_legged_robot.py` | IsaacGym | MRobot 私有 LeggedRobot base，负责 IsaacGym env 创建、terrain、asset、PD torque、domain randomization、reset、step、reward 调度、12 维 policy action 到 29 维 full action 的公共逻辑 |
| `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_env.py` | IsaacGym | BPM mimic 任务本体：加载 reference checkpoint、推进 BPM phase、生成 reference DOF/body state、构造 64/210 维观测、计算 mimic reward、输出 aux target |
| `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py` | IsaacLab/IsaacSim | IsaacLab DirectRLEnv 版本，不继承 IsaacGym base；复用同一套 MRobot 配置和 reference 语义，在 IsaacSim stage 中创建 articulation、contact sensor、obs/reward/reset/step |

文件之间的关系可以理解为：

```text
IsaacGym 训练:
train.py
  -> task_registry mrobot_music
  -> MrobotMimicEnv
  -> MrobotLeggedRobot
  -> MrobotBaseTask
  -> IsaacGym API

IsaacLab/IsaacSim 训练:
train_mrobot_isaaclab.py
  -> MrobotMimicIsaacLabEnv
  -> IsaacLab DirectRLEnv
  -> IsaacSim USD stage / PhysX

共享:
mrobot_mimic_config.py
reference_state.py
resources/robots/CASBOT02...
```

这样拆分的目的：

1. 保留旧仓库 MRobot IsaacGym 行为，方便和原训练逻辑对齐。
2. 不把 MRobot 的 12/29 action 特殊逻辑塞进 XBot 公共 base，避免影响现有任务。
3. IsaacLab 版本可以独立适配 IsaacSim 的 DirectRLEnv 生命周期。
4. 后续如果 MRobot IsaacLab 训练稳定后，可以逐步抽象公共逻辑，减少重复；当前优先保证迁移可读、可验证、低风险。

## 6. Reference checkpoint 格式要求

MRobot BPM mimic 需要一个外部 reference checkpoint。checkpoint 至少应包含：

1. `input_dim`
2. `output_dim`
3. `hidden`
4. `model_state_dict`
5. `output_columns`
6. `bpm_mean`
7. `bpm_std`
8. `target_mean`
9. `target_std`

输入编码固定为：

```text
[normalized_bpm, sin(phase_rad), cos(phase_rad)]
```

关节输出列名会通过 `JOINT_NAME_ALIASES` 映射到当前 URDF DOF。例如：

| reference 数据列前缀 | 当前 URDF joint |
| --- | --- |
| `left_leg_pelvic_pitch` | `leg_l1_joint` |
| `left_leg_pelvic_roll` | `leg_l2_joint` |
| `left_leg_pelvic_yaw` | `leg_l3_joint` |
| `left_leg_knee_pitch` | `leg_l4_joint` |
| `left_leg_ankle_pitch` | `leg_l5_joint` |
| `left_leg_ankle_roll` | `leg_l6_joint` |
| `right_leg_pelvic_pitch` | `leg_r1_joint` |
| `right_leg_pelvic_roll` | `leg_r2_joint` |
| `right_leg_pelvic_yaw` | `leg_r3_joint` |
| `right_leg_knee_pitch` | `leg_r4_joint` |
| `right_leg_ankle_pitch` | `leg_r5_joint` |
| `right_leg_ankle_roll` | `leg_r6_joint` |
| `waist_yaw` | `waist_yaw_joint` |

默认路径为：

```text
deploy/reference_state_keypoint_model.pt
```

该路径相对当前仓库根目录。推荐运行时直接传绝对路径：

```bash
--reference_model /abs/path/to/reference_state_keypoint_model.pt
```

## 6.5 IsaacSim、USD 和当前 URDF asset 的关系

IsaacSim/IsaacLab 最终运行时确实是在 USD stage 里创建和仿真对象。当前 MRobot 代码仓库里只有 URDF，是因为 IsaacLab 支持在运行时通过 `UrdfFileCfg` 把 URDF 转换成 USD，并把转换后的 articulation 放进 stage。

当前代码位置：

```python
robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=MrobotMimicCfg.asset.file.format(...),
        ...
    ),
)
```

运行流程是：

1. `MrobotMimicIsaacLabEnvCfg.robot.spawn` 指向 URDF 文件。
2. IsaacLab/IsaacSim 启动后调用 URDF importer/converter。
3. converter 读取 URDF、mesh、joint limit、inertia、collision、visual 等信息。
4. converter 临时生成或缓存 USD 表达，并在当前 USD stage 的 `/World/envs/env_*/Robot` 下生成 articulation prim。
5. 训练过程中 PhysX 和 IsaacLab tensor API 操作的是 stage 中的 USD articulation，而不是直接操作 URDF 文本。

所以现在“不提交 USD”也是可以跑 IsaacSim 的。区别是：

| 方案 | 当前状态 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 运行时 URDF -> USD | 当前 MRobot 使用 | 不需要额外维护 USD；URDF 改动后直接生效 | 第一次启动会有转换开销；不同 IsaacSim 版本的 URDF importer 行为可能略有差异 |
| 预先导出 USD 并用 `UsdFileCfg` 加载 | 当前未使用 | 启动更稳定、更快；asset 版本完全固定 | 需要单独维护 USD、mesh 路径和导出流程 |

本次迁移优先保持旧仓库 URDF asset 和关节命名不变，因此先采用运行时转换。后续如果要固定 IsaacSim 部署资产，可以新增：

1. 预转换的 robot USD asset。
2. `MrobotMimicIsaacLabEnvCfg.robot.spawn = sim_utils.UsdFileCfg(...)` 分支。
3. 文档记录 URDF -> USD 的导出命令、IsaacSim 版本和生成路径。

还有一个相关点：IsaacLab 默认 plane terrain 会引用 IsaacSim 的远程 `default_environment.usd`。为了避免无网络/受限网络环境失败，MRobot IsaacLab env 当前改成直接在 stage 中生成本地静态薄 cuboid 地面；这个地面也是 USD stage 里的 prim，只是不依赖外部 USD 文件。

## 7. 运行流程

### 7.0 Python 环境选择

当前机器上不要直接依赖默认 `python`。本次验证发现：

1. `/home/weil/anaconda3/bin/python` 是 base 环境，当前没有 `torch`，不适合跑本任务。
2. `/home/weil/anaconda3/envs/hl_rl/bin/python` 能加载 `torch` 和 reference checkpoint，适合跑 IsaacGym、reference 脚本、export、sim2sim。
3. `/home/weil/anaconda3/envs/humanoidgym/bin/python` 能导入 `isaaclab`，适合跑 IsaacLab/IsaacSim 训练和播放。
4. `/home/weil/anaconda3/envs/issac_sim/bin/python`、`/home/weil/anaconda3/envs/legged_lab/bin/python` 也能导入 `isaaclab`，如果你本地习惯用这两个环境，也可以切换后运行。

下面命令用绝对 Python 路径写出，避免误用 base 环境。

### 7.1 准备 reference checkpoint

如果已经有外部 reference checkpoint，直接记录路径：

```bash
REF=/home/weil/HumanoidGym-Ex/deploy/reference_state_keypoint_model.pt
```

当前已经验证过用户导入的 checkpoint：

| 文件 | 作用 |
| --- | --- |
| `deploy/reference_state_keypoint_model.pt` | PyTorch reference checkpoint，训练/env/sim2sim 使用 |
| `deploy/reference_state_keypoint_model.onnx` | reference network ONNX 版本，外部部署可用 |
| `deploy/reference_state_keypoint_model_metadata.json` | reference 导出元数据 |

验证到的 checkpoint 关键 shape：

| 字段 | 数值 |
| --- | ---: |
| `input_dim` | 3 |
| `output_dim` | 210 |
| `hidden` | `[256, 256, 256]` |
| `output_columns` | 210 |
| `bpm_mean` | `83.085556` |
| `bpm_std` | `51.564278` |

已用 `encode_bpm_phase()` 构造 `[normalized_bpm, sin(phase), cos(phase)]`，并通过 `ReferenceStateNet` 前向，输出 shape 为 `(1, 210)`。

如果需要重新训练 reference network：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py \
  --data-dir /abs/path/to/bpm_keypoint_csv_dir \
  --output /abs/path/to/reference_state_keypoint_model.pt \
  --epochs 1500 \
  --batch-size 1024
```

可选导出 ONNX：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py \
  --data-dir /abs/path/to/bpm_keypoint_csv_dir \
  --output /abs/path/to/reference_state_keypoint_model.pt \
  --onnx-output /abs/path/to/reference_state_keypoint_model.onnx
```

检查脚本帮助：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py --help
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/play_reference_state_network.py --help
```

### 7.2 IsaacGym smoke 训练

先用少量环境跑 1 个 iteration：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/train.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

如果只想验证某个固定 BPM：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/train.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --fixed_bpm 120 \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

### 7.3 IsaacGym 正式训练

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/train.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --num_envs 4096 \
  --max_iterations 53000 \
  --headless \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --run_name bpm_dance_v1
```

训练日志默认写入：

```text
logs/mrobot_mimic_May_music_BPM/<date>_<run_name>/
```

### 7.4 IsaacGym 播放/导出策略

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/play.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --load_run <run_dir_name> \
  --checkpoint <checkpoint_id> \
  --num_envs 1
```

如果 `--checkpoint -1`，会自动加载该 run 下最后一个 `model_*.pt`。

### 7.5 IsaacLab/IsaacSim smoke 训练

需要在安装了 IsaacLab/IsaacSim 的 Python 环境中运行：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --num_envs 16 \
  --max_iterations 1 \
  --num_steps_per_env 24 \
  --headless \
  --device cuda:0
```

正式训练可提高环境数量和 iteration：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --num_envs 1024 \
  --max_iterations 5000 \
  --num_steps_per_env 24 \
  --headless \
  --device cuda:0 \
  --run_name bpm_isaaclab_v1
```

### 7.6 IsaacLab/IsaacSim 播放

无 policy 的 zero-action smoke：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/play_mrobot_isaaclab.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

加载 TorchScript policy：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/play_mrobot_isaaclab.py \
  --task mrobot_music \
  --reference_model "$REF" \
  --policy /abs/path/to/policy_1.pt \
  --num_envs 1 \
  --device cuda:0
```

### 7.7 导出 ONNX policy

训练得到 `model_*.pt` 后，用 `deploy/export_actor.py` 导出 ONNX。推荐使用 checkpoint 直接导出：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python deploy/export_actor.py \
  --ckpt_path logs/mrobot_mimic_May_music_BPM/<run_dir>/model_<iter>.pt \
  -o deploy/casbot_mrobot_bpm.onnx
```

如果 checkpoint 中带 `obs_normalizer`，导出的 ONNX 会包含归一化，`sim2sim_mimic.py` 只需要传 raw 64 维 obs。

如果你已经有 JIT policy，也可以走兼容旧流程：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python deploy/export_actor.py \
  --jit_path /abs/path/to/policy.pt \
  --ckpt_path logs/mrobot_mimic_May_music_BPM/<run_dir>/model_<iter>.pt \
  -o deploy/casbot_mrobot_bpm.onnx
```

### 7.8 MuJoCo sim2sim mimic

查看帮助：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/sim2sim_mimic.py --help
```

使用 JIT policy：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/sim2sim_mimic.py \
  --load_model /abs/path/to/policy_1.pt \
  --reference_model "$REF" \
  --bpm 120
```

使用 ONNX policy：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/sim2sim_mimic.py \
  --load_model deploy/casbot_mrobot_bpm.onnx \
  --reference_model "$REF" \
  --bpm 120 \
  --terrain
```

如果缺少 MuJoCo 或 ONNX Runtime，脚本会在加载对应功能时提示缺失依赖。

## 7.9 MRobot 运行参数详细说明

### 7.9.1 IsaacGym 训练入口：`humanoid_gym_ex/scripts/train.py`

用途：使用 IsaacGym 后端训练 `mrobot_music`，这是最接近旧仓库 `/home/weil/hl_rl` 的训练路径。

推荐命令：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/train.py \
  --task mrobot_music \
  --reference_model /home/weil/HumanoidGym-Ex/deploy/reference_state_keypoint_model.pt \
  --num_envs 4096 \
  --max_iterations 53000 \
  --headless \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --run_name bpm_dance_v1
```

MRobot 常用参数：

| 参数 | 默认值 | 作用 | 训练时是否常指定 |
| --- | --- | --- | --- |
| `--task` | `humanoid_ppo` | 任务名；MRobot 必须指定为 `mrobot_music` | 必须 |
| `--reference_model` | CLI 默认为 `None`；配置默认 `deploy/reference_state_keypoint_model.pt` | BPM reference checkpoint 路径 | 已有默认；换模型时指定 |
| `--num_envs` | CLI 默认为 `None`；MRobot 配置默认 `4096` | 并行环境数量 | 常指定 |
| `--max_iterations` | CLI 默认为 `None`；MRobot PPO 默认 `53000` | PPO 更新次数 | 常指定 |
| `--seed` | CLI 默认为 `None`；MRobot PPO 默认 `5` | 随机种子 | 常指定 |
| `--run_name` | CLI 默认为 `None`；MRobot 配置默认空字符串 | 本次 run 名称，会拼到日志目录 | 常指定 |
| `--experiment_name` | CLI 默认为 `None`；MRobot 默认 `mrobot_mimic_May_music_BPM` | 日志实验目录名 | 偶尔指定 |
| `--headless` | `False` | 是否关闭 viewer；服务器训练建议打开 | 常指定 |
| `--rl_device` | `cuda:0` | PPO/神经网络训练设备 | 常指定 |
| `--sim_device` | IsaacGym 内置参数，建议显式传 `cuda:0` | IsaacGym 物理仿真设备 | 常指定 |
| `--fixed_bpm` | `None` | 固定 BPM；不传时按 `motion.bpm_range` 随机采样 | 调试时指定 |
| `--resume` | `False` | 是否从 checkpoint 继续训练 | 断点续训时指定 |
| `--load_run` | CLI 默认为 `None`；配置内有旧默认值 | 续训/播放时要加载的 run 目录名 | 续训时指定 |
| `--checkpoint` | CLI 默认为 `None`；配置默认 `-1` | checkpoint 编号；`-1` 表示最后一个 | 续训时指定 |
| `--terrain` | `None`，使用配置 `plane` | `plane/rough/heightfield/trimesh`；rough 等映射到旧 trimesh terrain | 地形实验时指定 |
| `--measure_heights` | `False` | 是否加入地形高度测量 | 粗糙地形时指定 |
| `--terrain_curriculum` | `False` | 是否打开 terrain curriculum | 粗糙地形时指定 |

MRobot 配置中的关键默认值：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `MrobotMimicCfg.env.num_envs` | `4096` | IsaacGym 默认训练环境数量 |
| `MrobotMimicCfg.env.episode_length_s` | `10` | episode 时长 |
| `MrobotMimicCfg.env.num_observations` | `64` | actor obs |
| `MrobotMimicCfg.env.num_privileged_obs` | `210` | critic obs |
| `MrobotMimicCfg.env.num_policy_actions` | `12` | policy 输出腿部动作 |
| `MrobotMimicCfg.env.num_actions` | `29` | 环境内部全身 DOF/action |
| `MrobotMimicCfg.motion.bpm_range` | `[60.0, 170.0]` | reset 时随机 BPM 范围 |
| `MrobotMimicCfg.motion.include_zero_bpm` | `True` | 允许采样 0 BPM 静止片段 |
| `MrobotMimicCfg.motion.sample_integer_bpm` | `True` | BPM 采样为整数 |
| `MrobotMimicCfg.motion.fixed_bpm` | `None` | 默认不固定 BPM |
| `MrobotMimicCfg.control.decimation` | `10` | 1000 Hz sim、100 Hz policy |
| `MrobotMimicCfg.control.action_scale` | `0.25` | action 到关节 residual target 的缩放 |
| `MrobotMimicCfg.control.use_ref_residual_target` | `True` | 受控腿关节使用 `ref_dof_pos + residual` |
| `MrobotMimicCfgPPO.runner.num_steps_per_env` | `24` | 每个 PPO iteration 每个 env 采样步数 |
| `MrobotMimicCfgPPO.runner.max_iterations` | `53000` | 默认最大 iteration |
| `MrobotMimicCfgPPO.runner.save_interval` | `500` | checkpoint 保存间隔 |
| `MrobotMimicCfgPPO.algorithm.learning_rate` | `1e-4` | PPO 学习率 |
| `MrobotMimicCfgPPO.algorithm.num_learning_epochs` | `5` | 每轮 PPO epoch |
| `MrobotMimicCfgPPO.algorithm.num_mini_batches` | `4` | mini-batch 数 |

### 7.9.2 IsaacGym 播放入口：`humanoid_gym_ex/scripts/play.py`

用途：加载已训练 checkpoint，在 IsaacGym 中播放/导出策略。

示例：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/play.py \
  --task mrobot_music \
  --reference_model /home/weil/HumanoidGym-Ex/deploy/reference_state_keypoint_model.pt \
  --load_run <run_dir_name> \
  --checkpoint -1 \
  --num_envs 1
```

常用参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--task` | `humanoid_ppo` | MRobot 播放必须指定 `mrobot_music` |
| `--reference_model` | 配置默认 `deploy/reference_state_keypoint_model.pt` | reference checkpoint |
| `--load_run` | CLI 默认为 `None` | 指定 `logs/mrobot_mimic_May_music_BPM/` 下的 run 目录 |
| `--checkpoint` | CLI 默认为 `None`；配置默认 `-1` | checkpoint 编号，`-1` 表示最后一个 |
| `--num_envs` | CLI 默认为 `None`；MRobot 配置默认 `4096` | 播放建议指定 `1` |
| `--fixed_bpm` | `None` | 指定播放时固定 BPM |
| `--headless` | `False` | 是否关闭 viewer |

### 7.9.3 IsaacLab/IsaacSim 训练入口：`train_mrobot_isaaclab.py`

用途：使用 IsaacLab DirectRLEnv 在 IsaacSim 中训练 MRobot BPM mimic。

推荐 smoke：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --headless \
  --num_envs 40 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name mrobot_bpm_isaaclab_1000 \
  --device cuda:0 \
  --reference_model deploy/reference_state_keypoint_model.pt
```

脚本自定义参数：

| 参数 | 默认值 | 作用 | 训练时是否常指定 |
| --- | --- | --- | --- |
| `--task` | `mrobot_music` | 兼容参数；MRobot IsaacLab 脚本是专用入口，不走 task registry，传错会报错 | 可指定 |
| `--num_envs` | `None`，不传则用 `MrobotMimicLabCfg.env.num_envs=4096` | IsaacLab 并行环境数量 | smoke/调参时常指定 |
| `--max_iterations` | `None`，不传则用 `MrobotMimicLabCfgPPO.runner.max_iterations=53000` | PPO iteration 数 | smoke/调参时常指定 |
| `--num_steps_per_env` | `None`，不传则用 `MrobotMimicLabCfgPPO.runner.num_steps_per_env=24` | 每轮每个 env 采样步数 | smoke/调参时常指定 |
| `--experiment_name` | `None`，不传则用 `MrobotMimicLabCfgPPO.runner.experiment_name=mrobot_mimic_May_music_BPM_isaaclab` | 日志实验目录名 | 想区分 IsaacLab/IsaacGym 时指定 |
| `--reference_model` | `None`，不传则用配置默认 `deploy/reference_state_keypoint_model.pt` | reference checkpoint | 已有默认；换模型时指定 |
| `--seed` | `None`，不传则用 `MrobotMimicLabCfgPPO.seed=5` | 随机种子 | 常指定 |
| `--run_name` | `None`，不传则用 `MrobotMimicLabCfgPPO.runner.run_name=''` | 日志 run 名称 | 常指定 |
| `--no_log` | `False` | 不创建 log_dir；通常也不会保存 checkpoint | smoke 时可指定 |
| `--disable_domain_randomization` | `False` | 关闭 IsaacLab MRobot 域随机化，便于排查是否随机化导致不稳定 | debug 时指定 |
| `--deterministic_reset` | `False` | 关闭 reset 姿态扰动、root xy/yaw 扰动；BPM 采样仍按 motion 配置 | debug 时指定 |

如果不传 `--no_log`，脚本会在 log 目录中复制 `MrobotMimicCfgPPO.runner.save_config` 指定的配置文件：

```text
mrobot_mimic_config_lab.py
```

注意：`save_config` 只是一个“要保存哪份配置文件”的字段，PPO runner 本身不会自动读取它；IsaacLab 专用训练入口现在显式复制该文件。

IsaacLab `AppLauncher` 参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--headless` | `False` | 关闭窗口/显示，服务器训练建议指定 |
| `--device` | 由 IsaacLab AppLauncher 决定；建议显式指定 `cuda:0` | IsaacSim/IsaacLab 仿真设备 |
| `--livestream` | IsaacLab 默认 | 远程直播模式，取值 `0/1/2` |
| `--enable_cameras` | `False` | 是否启用相机相关扩展 |
| `--experience` | IsaacLab 根据 headless 自动选择 | 指定 `.kit` experience 文件 |
| `--rendering_mode` | IsaacLab 默认 | `quality/balanced/performance` |
| `--kit_args` | 空 | 透传给 Omniverse Kit 的额外参数 |

IsaacLab 版本没有复用 IsaacGym 的 `--fixed_bpm` CLI。如果要固定 BPM，当前做法是在代码或配置中设置 `MrobotMimicCfg.motion.fixed_bpm`，后续可以按需要给 `train_mrobot_isaaclab.py` 增加同名参数。

### 7.9.4 IsaacLab/IsaacSim 播放入口：`play_mrobot_isaaclab.py`

用途：在 IsaacSim 中加载 TorchScript policy 播放；不传 policy 时用 zero action 做 env smoke。

示例：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/play_mrobot_isaaclab.py \
  --task mrobot_music \
  --headless \
  --device cuda:0 \
  --num_envs 1 \
  --steps 1200 \
  --reference_model deploy/reference_state_keypoint_model.pt \
  --policy /abs/path/to/policy_1.pt
```

参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--task` | `mrobot_music` | 兼容参数；播放脚本固定创建 MRobot env |
| `--num_envs` | `1` | 播放环境数量 |
| `--policy` | `None` | TorchScript policy 路径；不传则 zero action |
| `--reference_model` | `None`，不传则用配置默认路径 | reference checkpoint |
| `--steps` | `1200` | 播放 step 数 |
| `--seed` | `5` | 随机种子 |
| `--headless` | `False` | AppLauncher 参数，是否关闭显示 |
| `--device` | IsaacLab AppLauncher 默认；建议显式 `cuda:0` | 仿真设备 |

### 7.9.5 ONNX 导出入口：`deploy/export_actor.py`

用途：把训练得到的 `model_*.pt` 导出成 sim2sim 使用的 ONNX policy。

推荐：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python deploy/export_actor.py \
  --ckpt_path logs/mrobot_mimic_May_music_BPM/<run_dir>/model_<iter>.pt \
  -o deploy/casbot_mrobot_bpm.onnx
```

参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--ckpt_path` | `None` | 训练 checkpoint；推荐只传这个即可直接导出 ONNX |
| `--jit_path` | `None` | 兼容旧流程，传已导出的 TorchScript policy |
| `--output` / `-o` | `deploy/casbot_mimic.onnx` | 输出 ONNX 路径 |

导出逻辑：

1. 从当前 `MrobotMimicCfg` / `MrobotMimicCfgPPO` 读取 actor 输入输出维度。
2. 重建 `ActorCritic`。
3. 加载 checkpoint 的 `model_state_dict`。
4. 如果 checkpoint 含 `obs_normalizer`，把 normalizer baked 到 ONNX。
5. 用 dummy `(1, 64)` obs 做前向检查，输出应为 `(1, 12)`。

### 7.9.6 MuJoCo sim2sim：`sim2sim_mimic.py`

用途：加载 ONNX/JIT policy，在 MuJoCo 中做 MRobot BPM mimic sim2sim。

ONNX 示例：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/sim2sim_mimic.py \
  --load_model deploy/casbot_mrobot_bpm.onnx \
  --reference_model deploy/reference_state_keypoint_model.pt \
  --bpm 120 \
  --terrain
```

参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--load_model` | 必填 | `.onnx` 或 TorchScript `.pt` policy |
| `--reference_model` | `{LEGGED_GYM_ROOT_DIR}/deploy/reference_state_keypoint_model.pt` | reference checkpoint |
| `--bpm` | `70.0` | 前 1000 个低层 step 后使用的 BPM |
| `--terrain` | `False` | 不传为 plane MuJoCo XML；传入后用 terrain XML |

### 7.9.7 Reference network 脚本

训练 reference network：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py \
  --data-dir /abs/path/to/bpm_keypoint_csv_dir \
  --output /abs/path/to/reference_state_keypoint_model.pt \
  --epochs 1500 \
  --batch-size 1024
```

主要参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-dir` | 脚本内 `DEFAULT_DATA_DIR` | 包含 `bpm_*_keypoint.csv` 的目录 |
| `--output` / `-o` | 脚本内 `DEFAULT_OUTPUT_PATH` | 输出 `.pt` checkpoint |
| `--onnx-output` | `None`，默认使用 output 同名 `.onnx` | 输出 ONNX |
| `--no-onnx` | `False` | 不导出 ONNX |
| `--epochs` | `1500` | 训练 epoch |
| `--batch-size` | `1024` | batch size |
| `--bpm-values` | `0,60:170` | 读取哪些 BPM 文件 |
| `--static-repeat` | `20` | 静态 `bpm_000` 重复次数 |
| `--static-loss-weight` | `20` | 静态样本 loss 权重 |
| `--lr` | `1e-3` | Adam 学习率 |
| `--hidden` | `256,256,256` | MLP hidden size |
| `--val-fraction` | `0.1` | 验证集比例 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | `auto/cpu/cuda` |
| `--log-every` | `100` | 每多少 epoch 打印一次 |

播放/检查 reference network：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/play_reference_state_network.py \
  --model deploy/reference_state_keypoint_model.pt \
  --bpm 120 \
  --output-dir reference_state_eval
```

主要参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model` | 脚本内 `DEFAULT_MODEL_PATH` | reference checkpoint |
| `--data-dir` | 脚本内 `DEFAULT_DATA_DIR` | 用于查找验证 CSV |
| `--bpm` | `90.0` | 要评估/生成的 BPM |
| `--csv` | `None` | 显式指定验证 CSV |
| `--phase` | `None` | 指定单个相位，单位 rad |
| `--output-dir` / `-o` | `reference_state_eval` | 输出 plot/metrics/prediction CSV |
| `--num-phases` | `301` | 没有 CSV 时采样多少个 phase |
| `--joints` | 空字符串 | 指定要画的 joint base name |
| `--device` | `auto` | `auto/cpu/cuda` |

## 7.10 任务选择、关节顺序和 ONNX 顺序

### 7.10.1 IsaacLab 训练是否需要 `--task mrobot_music`

IsaacGym 的通用入口 `humanoid_gym_ex/scripts/train.py` 必须指定：

```bash
--task mrobot_music
```

原因是 `train.py` 通过 `task_registry` 根据 task name 创建环境和 PPO runner。

MRobot IsaacLab 入口 `humanoid_gym_ex/scripts/train_mrobot_isaaclab.py` 是专用脚本，代码里直接创建：

```python
direct_env = MrobotMimicIsaacLabEnv(env_cfg)
```

所以它不依赖 task registry，理论上不需要 `--task`。为了命令风格统一，脚本现在接受：

```bash
--task mrobot_music
```

这个参数只是兼容和防误用，不参与环境选择；如果传其他 task 会直接报错。

### 7.10.2 IsaacGym 和 IsaacSim 关节顺序不一致如何处理

是的，IsaacGym 和 IsaacSim/IsaacLab 从 URDF/USD 解析 articulation 时，关节顺序可能不一致。一个常见差异是 IsaacGym 更接近深度优先，IsaacSim/IsaacLab 可能更接近广度优先或 importer 自己的排序。MRobot 代码不依赖 IsaacLab 的原始 joint order，而是建立了“规范关节顺序 canonical order”。

规范顺序来自：

```python
MrobotMimicCfg.init_state.default_joint_angles.keys()
```

当前 29 DOF 规范顺序是：

| index | joint |
| ---: | --- |
| 0 | `leg_l1_joint` |
| 1 | `leg_l2_joint` |
| 2 | `leg_l3_joint` |
| 3 | `leg_l4_joint` |
| 4 | `leg_l5_joint` |
| 5 | `leg_l6_joint` |
| 6 | `leg_r1_joint` |
| 7 | `leg_r2_joint` |
| 8 | `leg_r3_joint` |
| 9 | `leg_r4_joint` |
| 10 | `leg_r5_joint` |
| 11 | `leg_r6_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `upper_left_1_joint` |
| 14 | `upper_left_2_joint` |
| 15 | `upper_left_3_joint` |
| 16 | `upper_left_4_joint` |
| 17 | `upper_left_5_joint` |
| 18 | `upper_left_6_joint` |
| 19 | `upper_left_7_joint` |
| 20 | `upper_right_1_joint` |
| 21 | `upper_right_2_joint` |
| 22 | `upper_right_3_joint` |
| 23 | `upper_right_4_joint` |
| 24 | `upper_right_5_joint` |
| 25 | `upper_right_6_joint` |
| 26 | `upper_right_7_joint` |
| 27 | `vhead_1_joint` |
| 28 | `vhead_2_joint` |

受控关节是前 12 个腿部关节：

```python
MrobotMimicCfg.env.num_control = [0, 1, ..., 11]
```

IsaacLab 中的顺序适配位置在 `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`：

```python
self.canonical_joint_names = list(cfg.init_state.default_joint_angles.keys())
joint_sim_ids = [self.robot.joint_names.index(name) for name in self.canonical_joint_names]
self.joint_sim_ids = torch.tensor(joint_sim_ids, ...)
```

读取 IsaacLab joint tensor 时：

```python
self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
```

也就是说，进入 MRobot obs/reward/action 逻辑前，IsaacLab 的 sim-order 已经被重排成 MRobot 规范顺序。

写回 IsaacLab torque 时再反向映射：

```python
sim_order_torques = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
sim_order_torques[:, self.joint_sim_ids] = self.torques
self.backend.set_dof_targets(sim_order_torques)
```

IsaacGym 版本在 `humanoid_gym_ex/envs/robots/mrobot/mrobot_legged_robot.py` 中使用：

```python
self.dof_names = self.gym.get_asset_dof_names(robot_asset)
```

旧 IsaacGym MRobot 任务默认认为 asset DOF order 与旧训练顺序一致；reference 解码仍然通过关节名做映射，不按 reference 输出顺序硬切。

reference checkpoint 的关节列顺序也不直接信任，而是按列名解码：

```python
JOINT_NAME_ALIASES
self.reference_column_index
self.ref_dof_pos_indices
self.ref_dof_vel_indices
```

IsaacGym 对应位置：

```text
humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_env.py::_build_dof_column_indices()
```

IsaacLab 对应位置：

```text
humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py::_init_reference_network()
```

### 7.10.3 ONNX 在 IsaacGym 和 IsaacLab 训练下顺序是否一致

一致。`deploy/export_actor.py` 只导出 actor MLP，本身不携带关节重排逻辑；顺序一致依赖训练环境给 actor 的 64 维 obs 和 12 维 action 都使用同一套 MRobot 规范顺序。

两种训练后导出的 ONNX 都约定：

```text
input:  obs, shape = [N, 64]
output: actions, shape = [N, 12]
```

12 维 ONNX 输出动作顺序是：

| action index | joint |
| ---: | --- |
| 0 | `leg_l1_joint` |
| 1 | `leg_l2_joint` |
| 2 | `leg_l3_joint` |
| 3 | `leg_l4_joint` |
| 4 | `leg_l5_joint` |
| 5 | `leg_l6_joint` |
| 6 | `leg_r1_joint` |
| 7 | `leg_r2_joint` |
| 8 | `leg_r3_joint` |
| 9 | `leg_r4_joint` |
| 10 | `leg_r5_joint` |
| 11 | `leg_r6_joint` |

64 维 ONNX 输入 obs 顺序是：

| range | 维度 | 内容 |
| --- | ---: | --- |
| `0:12` | 12 | 腿部关节位置误差 `q[:, num_control]` |
| `12:24` | 12 | 腿部关节速度 `dq[:, num_control]` |
| `24:36` | 12 | 上一帧/当前 policy action，腿部 12 维 |
| `36:39` | 3 | base angular velocity |
| `39:42` | 3 | base euler xyz |
| `42:43` | 1 | `sin(phase_rad)` |
| `43:44` | 1 | `cos(phase_rad)` |
| `44:45` | 1 | normalized BPM command |
| `45:57` | 12 | reference 腿部关节位置 |
| `57:58` | 1 | reference waist/root z |
| `58:60` | 2 | reference waist roll + pitch |
| `60:63` | 3 | reference waist linear velocity |
| `63:64` | 1 | reference waist angular velocity z |

IsaacGym obs 构造位置：

```text
humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_env.py::compute_observations()
```

IsaacLab obs 构造位置：

```text
humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py::_get_observations()
```

sim2sim 使用同样顺序接收 ONNX 输出：

```python
raw_action[cfg.env.num_control] = rl_out
```

位置：

```text
humanoid_gym_ex/scripts/sim2sim_mimic.py
```

### 7.10.4 `isaaclab_env.py` 是否等于 `mrobot_mimic_env.py + mrobot_legged_robot.py`

概念上可以这样理解：`isaaclab_env.py` 是 IsaacLab/IsaacSim 后端下的“合并版 MRobot 任务”，它承担了 IsaacGym 版本里 `mrobot_legged_robot.py` 和 `mrobot_mimic_env.py` 的主要职责。

对应关系：

| IsaacGym 文件 | 主要职责 | IsaacLab 中对应 |
| --- | --- | --- |
| `mrobot_legged_robot.py` | 创建 sim/env/asset、维护 DOF/root/contact buffer、PD torque、action 扩展、reset、domain randomization、reward 调度 | `isaaclab_env.py` 的 `_setup_scene()`、`_init_buffers()`、`_pre_physics_step()`、`_apply_action()`、`_reset_idx()`、`_prepare_reward_function()` |
| `mrobot_mimic_env.py` | BPM reference、phase、obs、privileged obs、mimic reward、aux target | `isaaclab_env.py` 的 `_init_reference_network()`、`compute_ref_state()`、`_get_observations()`、reward functions |

但它不是简单复制粘贴，也不是 100% 等价：

1. IsaacLab 的生命周期是 `DirectRLEnv`，不是 IsaacGym `BaseTask/LeggedRobot`。
2. IsaacLab 机器人状态来自 `Articulation.data`，不是 IsaacGym root/dof tensor API。
3. IsaacLab 写 action 用 `set_joint_effort_target()`，需要先从 MRobot 规范顺序映射回 IsaacLab sim-order。
4. IsaacLab 版本已经接入主要 domain randomization，并已补齐 sys delay、ankle actor obs、ankle PD delay/filter/noise 链路；contact offset/rest offset 和 euler bias 仍未完全等价。
5. IsaacLab 版本为了离线稳定，plane terrain 使用本地 USD prim，而不是远程 grid USD。

### 7.10.5 IsaacLab 训练时哪些配置生效

IsaacLab 专用入口会同时使用两类配置：

1. `MrobotMimicLabCfg`：继承 `MrobotMimicCfg`，用于 IsaacLab 环境、机器人、reference、观测、奖励和 domain randomization。
2. `MrobotMimicLabCfgPPO`：继承 `MrobotMimicCfgPPO`，用于 IsaacLab PPO policy/algorithm/runner 默认值。

已经生效的主要配置：

| 配置区域 | IsaacLab 中是否生效 | 说明 |
| --- | --- | --- |
| `env.num_observations` | 生效 | `MrobotMimicIsaacLabEnvCfg.observation_space=64` |
| `env.num_privileged_obs` | 生效 | `state_space=210` |
| `env.num_policy_actions` | 生效 | `action_space=12` |
| `env.num_actions` | 生效 | 内部 full DOF/action buffer 为 29 |
| `env.num_control/num_notcontrol/ref_num_notcontrol` | 生效 | 控制腿部 12 维，非控制关节跟 reference |
| `env.num_envs` | 生效 | `train_mrobot_isaaclab.py` 不传 `--num_envs` 时使用 4096 |
| `env.episode_length_s` | 生效 | `DirectRLEnvCfg.episode_length_s` |
| `motion.reference_model_path` | 生效 | 默认 `deploy/reference_state_keypoint_model.pt` |
| `motion.bpm_range/include_zero_bpm/sample_integer_bpm/fixed_bpm/init_phase_range` | 生效 | reset 时采样 BPM/phase |
| `asset.file/fix_base_link/self_collisions` | 生效 | 通过 IsaacLab `UrdfFileCfg` 导入 |
| `init_state.pos/default_joint_angles` | 生效 | IsaacLab articulation 初始默认姿态使用该配置；reset 训练状态会再按 BPM reference 当前帧覆盖 29 DOF |
| `control.decimation/action_scale/stiffness/damping/use_ref_residual_target` | 生效 | policy rate、action scale、PD gain、受控关节 target 语义使用这些值 |
| `lab_joint_effort_limits/lab_joint_velocity_limits/lab_joint_position_limits` | 生效 | IsaacLab 专用显式 limit；actuator effort/velocity 和内部 torque limit 都从这里读取 |
| `sim.dt/physx solver/contact pair 等基础项` | 生效 | 映射到 `SimulationCfg` / `PhysxCfg` |
| `terrain.static_friction/dynamic_friction/restitution` | 生效 | 用在本地 plane physics material；每 env 材质随机化见 `domain_rand` |
| `rewards.scales` 和 `rewards.sigma` | 生效一部分 | 当前 IsaacLab env 已实现的 reward 会按这些 scale/sigma 计算 |
| `normalization.obs_scales/clip_actions` | 生效 | obs 缩放和 action clip 使用这些值 |
| `policy.actor_hidden_dims/critic_hidden_dims/init_noise_std` | 生效 | PPO actor-critic 使用 |
| `algorithm.learning_rate/gamma/lam/entropy_coef/num_learning_epochs/num_mini_batches` | 生效 | PPO 使用 |
| `runner.max_iterations/num_steps_per_env/save_interval/experiment_name/run_name/save_config` | 生效 | CLI 不传时使用 `MrobotMimicLabCfgPPO` 默认；`save_config` 会复制到 log 目录 |

尚未完全生效或只保留占位的配置：

| 配置区域 | 当前状态 | 原因 |
| --- | --- | --- |
| `env.num_aux` | 未用于 IsaacLab PPO aux loss | 当前 `isaaclab_env.py` 没有返回 aux target；IsaacGym 版本返回 aux |
| `env.normalize_obs` | 生效 | `IsaacLabRslRlVecEnv` 对 MRobot 暴露 `cfg.env.normalize_obs`，PPO runner 会创建 actor/critic running mean/std normalizer；IsaacLab 版暂不启用 aux loss |
| `terrain.curriculum/measure_heights/rough terrain` | 未完整接入 MRobot IsaacLab | 当前 MRobot IsaacLab 先支持本地 plane |
| `domain_rand.randomize_kp/kd/motor_strength/motor_offset/default_dof_pos_offset` | 生效 | reset 时按 range 采样，`_apply_action()` 计算 torque/target 时使用 |
| `domain_rand.randomize_init_dof_pos/init_dof_pos_range` | 生效 | reset 到 reference DOF 后再加小扰动；`--deterministic_reset` 时关闭 |
| `domain_rand.randomize_friction/static_friction_range/dynamic_friction_range/randomize_restitution` | 生效 | 写入 IsaacLab `root_physx_view.set_material_properties()` |
| `domain_rand.randomize_payload_mass/randomize_link_mass` | 生效 | 写入 `set_masses()`，并按质量比例同步 inertia |
| `domain_rand.randomize_com_displacement/com_*_pos_range` | 生效 | 写入 `set_coms()` |
| `domain_rand.randomize_joint_armature/joint_armature_values/joint_armature_range` | 生效 | 写入 `write_joint_armature_to_sim()`；不随机时初始化写入旧 IsaacGym 固定 armature values，后续 reset 不重复写 |
| `domain_rand.randomize_joint_friction/joint_friction_range` | 生效 | 写入 `write_joint_friction_coefficient_to_sim()`；不随机时初始化写入 0，后续 reset 不重复写 |
| `domain_rand.push_robots/disturbance` | 生效 | push 通过 root velocity 冲击；disturbance 通过 `set_external_force_and_torque()` 注入基座外力 |
| `domain_rand.action_delay/action_delay_range` | 生效 | full 29 DOF scaled action 在 physics substep 内进入 delay buffer |
| `noise.noise_scales` | 生效 | IsaacLab env 已按 IsaacGym 版 45 维 actor proprio 观测填充 `noise_scale_vec`；只对 45 维本体观测加噪声，不对 19 维 reference goal 加噪声 |
| `control.use_ref_residual_target` | 生效 | True 时 `ref_dof_pos + residual`；False 时 `default_dof_pos + default_dof_pos_offset + residual` |
| `runner.resume/load_run/checkpoint/resume_path` | 当前 IsaacLab 专用脚本未接入 | `train_mrobot_isaaclab.py` 还没有 resume CLI/加载逻辑 |
| `domain_rand.sys_delay`、`randomize_ankle_obs_*`、`randomize_ankle_pd*`、`randomize_ankle_motor_offset` | 生效 | 已补齐 actor observation delay/filter/bias、脚踝 PD 速度反馈 delay/filter/noise、脚踝 PD/motor offset reset 采样 |
| `randomize_euler_*`、`randomize_upperbody_speed` | 未完整接入 | euler bias 和旧 upperbody buffer 播放速度不是当前 BPM reference 主链路 |
| 旧 IsaacGym 低层 contact offset/rest offset randomization | 未等价 | IsaacLab/PhysX shape offset API 可写，但 MRobot 当前未作为训练随机化打开 |

因此现在的状态已经不是“IsaacSim 不能做域随机化”。IsaacLab 版已经把能通过 `Articulation`/`root_physx_view` 和控制张量直接写入的项接上了；剩下主要是 euler 观测偏置、旧 upperbody 速度播放和少量低层 contact offset 细节。

### 7.10.6 IsaacLab 观测归一化、观测噪声和 domain randomization 当前细节

IsaacLab MRobot 训练现在有两层观测处理：

1. 环境内部观测噪声：由 `MrobotMimicCfg.noise.add_noise`、`noise.noise_level` 和 `noise.noise_scales` 控制，在 `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py` 的 `_get_noise_scale_vec()` 和 `_get_observations()` 中执行。
2. PPO running mean/std 归一化：由 `MrobotMimicCfg.env.normalize_obs` 控制，在 `humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py` 暴露给 `OnPolicyRunner`，runner 会对 actor obs 和 critic obs 分别做 running mean/std。

`noise_scales` 的 45 维对应关系如下：

| 维度范围 | 内容 | 是否加噪声 |
| --- | --- | --- |
| `0:12` | 12 个控制关节的 `q - q_ref` | 使用 `noise_scales.dof_pos * obs_scales.dof_pos` |
| `12:24` | 12 个控制关节速度 | 使用 `noise_scales.dof_vel * obs_scales.dof_vel` |
| `24:36` | 上一步 12 维 policy action | 不加噪声 |
| `36:39` | base angular velocity | 使用 `noise_scales.ang_vel * obs_scales.ang_vel` |
| `39:42` | base roll/pitch/yaw | 使用 `noise_scales.euler` |
| `42:45` | `sin(phase), cos(phase), normalized_bpm` | 不加噪声 |

19 维 reference goal 不加环境噪声，保持和旧 IsaacGym mimic actor obs 语义一致。最终 actor obs 仍是 `45 + 19 = 64` 维。

IsaacLab 当前已经真正影响训练动力学或 reset 的 `domain_rand`：

| 配置 | 作用位置 | 说明 |
| --- | --- | --- |
| `randomize_kp/kp_range` | `_randomize_reset_buffers()`、`_apply_action()` | 每个 env reset 时采样 29 DOF 的 Kp 系数，torque 中乘到 `p_gains` |
| `randomize_kd/kd_range` | `_randomize_reset_buffers()`、`_apply_action()` | 每个 env reset 时采样 29 DOF 的 Kd 系数，torque 中乘到 `d_gains` |
| `randomize_motor_strength/motor_strength_range` | `_randomize_reset_buffers()`、`_apply_action()` | 每个 env reset 时采样 motor strength，torque 计算后整体相乘 |
| `randomize_motor_offset/motor_offset_range` | `_randomize_reset_buffers()`、`_apply_action()` | 每个 env reset 时采样 position target offset，加到 reference/residual target 上 |
| `randomize_default_dof_pos_offset` | `_randomize_reset_buffers()`、`_apply_action()` | `control.use_ref_residual_target=False` 时进入 default target；True 时保持旧 mimic 语义，不使用它 |
| `randomize_init_dof_pos/init_dof_pos_range` | `_reset_idx()` | 每次 reset 先对齐 reference DOF，再给 29 个 canonical DOF 初始位置加扰动 |
| `randomize_friction/static_friction_range/dynamic_friction_range` | `_randomize_materials()` | 写入 IsaacLab material properties |
| `randomize_restitution/restitution_range` | `_randomize_materials()` | 写入 IsaacLab material properties |
| `randomize_payload_mass/payload_mass_range` | `_randomize_mass_and_com()` | 写入 link mass，并同步 inertia |
| `randomize_link_mass/link_mass_range` | `_randomize_mass_and_com()` | 对非 payload 质量按比例缩放，并同步 inertia |
| `randomize_com_displacement/com_*_pos_range` | `_randomize_mass_and_com()` | 写入指定 body 的 COM |
| `randomize_joint_armature/joint_armature_values/joint_armature_range` | `_randomize_joint_physx_props()` | 写入 joint armature；默认不随机但写固定 armature |
| `randomize_joint_friction/joint_friction_range` | `_randomize_joint_physx_props()` | 写入 joint friction |
| `push_robots` | `_post_physics_step_callback()`、`_push_robots()` | 按 root velocity 注入速度冲击 |
| `disturbance` | `_post_physics_step_callback()`、`_disturbance_robots()` | 对 base body 设置一个持续一个 control step 的外力 |
| `action_delay/action_delay_range` | `_apply_action()` | 在 physics substep 中延迟 full 29 DOF scaled action |
| `sys_delay/imu_delay_range/motor_delay_range` | `_init_sys_delay_buffers()`、`_record_sys_delay_state()`、`_get_observations()` | actor obs 使用延迟后的 root state 和 q/dq |
| `randomize_ankle_obs_*` | `_resample_ankle_obs_bias()`、`_apply_actor_ankle_obs_bias()` | 只污染 actor 输入，不改真实状态和 critic |
| `randomize_ankle_pd*` | `_resample_ankle_dq_randomization()`、`_get_ankle_dq_for_pd()` | 只影响 PD 阻尼项中的脚踝 dq |
| `randomize_ankle_motor_offset` | `_randomize_reset_buffers()` | reset 时对指定脚踝关节采样 motor offset |

IsaacLab 当前仍未完整接入的 `domain_rand`：

| 配置 | 当前状态 |
| --- | --- |
| `randomize_euler_xy_offset`、`randomize_euler_z_offset` | 当前 actor obs 直接使用 base euler，没有额外 euler bias |
| `randomize_upperbody_speed` | BPM reference network 驱动上肢，不再使用旧 upperbody buffer 播放速度 |
| `randomize_contact_offsets/contact_offset_range/rest_offset_range` | IsaacLab 可通过 shape offset API 写入，但当前 MRobot Lab 训练未打开这项 |

### 7.10.7 IsaacLab 初期疯狂抽搐和短回合的修复点

这次重点排查了 IsaacLab 能跑但训练回合只有四五十步、reward 到负几百、可视化疯狂抽搐的问题。主要不是频率配置错了，`sim.dt=0.001`、`control.decimation=10`，所以 policy dt 仍是 `0.01s`，和 IsaacGym 版 100Hz 控制频率一致。更关键的差异在 reset 和 PD 控制语义：

| 问题 | 原状态 | 已修复位置 |
| --- | --- | --- |
| reset 没有初始化到 reference 姿态 | IsaacLab 版 reset 直接使用 `robot.data.default_joint_pos`，随后 PD 立刻追 BPM reference，初始误差过大，容易大力矩抽搐 | `isaaclab_env.py::_reset_idx()` 现在先采样 BPM/phase，再 `compute_ref_state()`，然后把 29 DOF 写到 reference 当前帧 |
| decimation 内 PD 使用旧状态 | `_apply_action()` 在 10 个 physics substep 内重复使用同一份旧 `dof_pos/dof_vel` | `_apply_action()` 每个 substep 都从 `robot.data.joint_pos/joint_vel` 重读最新状态 |
| torque limit 过大且不分关节 | 原 IsaacLab actuator 把所有关节 `effort_limit_sim=250`，上肢/头部会比 URDF 限幅大很多 | actuator 改为使用 URDF effort limit；内部 `torque_limits` 从 IsaacLab joint effort limit 读取 |
| reference 或 action target 可能越过 URDF limit | 左 wrist roll 默认/参考可能低于 URDF lower，IsaacLab 对 limit 更严格 | `compute_ref_state()` 和 `_apply_action()` 都会把 reference/target clamp 到 `joint_pos_limits` |
| 旧 IsaacGym actions_filter 没接 | IsaacGym 在 decimation 内从上一次 full action 插值到当前 full action | IsaacLab `_apply_action()` 已按 `normalization.actions_filter` 做 substep 插值 |
| action delay 没接 | 旧 Gym 在 `_compute_torques()` 内对 scaled full action 做 delay | IsaacLab 已用 `action_delay_buffer` 对 full 29 DOF scaled action 做 delay |

建议调试流程：

1. 先关闭域随机化和 reset 噪声确认基础控制稳定：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --headless \
  --num_envs 40 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name mrobot_lab_no_dr_debug \
  --device cuda:0 \
  --disable_domain_randomization \
  --deterministic_reset
```

2. 如果基础稳定，再去掉 `--deterministic_reset`，只保留 `--disable_domain_randomization`。
3. 最后打开完整域随机化，也就是两个 debug 参数都不传。
4. 如果还是出现 40-50 step 大量 reset，优先看 `logs/.../episode` 里的 termination、root height、torque limit 和 action rate，而不是先调 PPO 参数。

## 8. 常用验证命令

检查 task registry：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python -c "import humanoid_gym_ex.envs as envs; reg=envs.register_tasks(); print(sorted(reg.task_classes.keys()))"
```

预期输出包含：

```text
['humanoid_ppo', 'mrobot_music']
```

检查 MRobot config shape：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python -c "from humanoid_gym_ex.envs.robots.mrobot.mrobot_mimic_config import MrobotMimicCfg; print(MrobotMimicCfg.env.num_observations, MrobotMimicCfg.env.num_privileged_obs, MrobotMimicCfg.env.num_actions, MrobotMimicCfg.env.num_policy_actions, MrobotMimicCfg.env.num_aux)"
```

预期：

```text
64 210 29 12 9
```

检查 PPO aux / no-aux 都能跑：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python -c "from humanoid_gym_ex.algo.ppo.actor_critic import ActorCritic; from humanoid_gym_ex.algo.ppo.ppo import PPO; import torch; ac=ActorCritic(64,210,12,num_aux=9); alg=PPO(ac); alg.init_storage(2,3,[64],[210],[12],[9]); obs=torch.randn(2,64); critic=torch.randn(2,210); alg.act(obs,critic); alg.process_env_step(torch.ones(2), torch.zeros(2,dtype=torch.bool), {}, torch.randn(2,9)); alg.compute_returns(critic); print(alg.update())"
```

检查脚本帮助：

```bash
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/train.py --help
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/play.py --help
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/sim2sim_mimic.py --help
/home/weil/anaconda3/envs/hl_rl/bin/python humanoid_gym_ex/scripts/bpm/train_reference_state_network.py --help
/home/weil/anaconda3/envs/hl_rl/bin/python deploy/export_actor.py --help
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py --help
```

## 9. 当前验证结果

已在当前环境完成：

1. 新增/修改 Python 文件 compileall 通过。
2. `sim2sim_mimic.py --help` 通过。
3. `train_reference_state_network.py --help` 通过。
4. `train.py --help` 通过。
5. `play.py --help` 通过。
6. MRobot config shape 检查通过：`64 210 29 12 9`。
7. `envs.register_tasks()` 可注册 `humanoid_ppo` 和 `mrobot_music`。
8. PPO 带 aux 和不带 aux 的轻量 update 都通过。
9. `deploy/reference_state_keypoint_model.pt` 已验证：输入 `(1, 3)`，输出 `(1, 210)`。
10. `deploy/export_actor.py --help` 通过，并已适配当前仓库 actor/config/normalizer。
11. 使用临时 dummy checkpoint 实测 `deploy/export_actor.py --ckpt_path /tmp/mrobot_dummy_ckpt.pt -o /tmp/mrobot_dummy.onnx` 通过，前向输出维度为 `(1, 12)`。
12. `/home/weil/anaconda3/envs/humanoidgym/bin/python -c "import isaaclab"` 通过，说明 IsaacLab 环境存在。
13. `train_mrobot_isaaclab.py --help` 通过，包含 `--disable_domain_randomization` 和 `--deterministic_reset`。
14. 本轮修改后的 `isaaclab_env.py`、Lab/Gym config、`train_mrobot_isaaclab.py` 语法检查通过。
15. 当前工具环境尝试了极小 MRobot IsaacLab smoke；`--device cuda:0` 因 `No CUDA GPUs are available` 退出，`--device cpu` 已越过 AppLauncher、Base environment、URDF importer，但最终仍因当前机器没有可用 GPU/图形后端报 `no suitable CUDA GPU was found`，没有完成 1 iteration。

未在当前沙箱完成的验证：

1. IsaacGym 真正创建 MRobot env 的 smoke step，因为当前工具环境没有可用 CUDA 设备。
2. IsaacLab/IsaacSim 完整 1 iteration smoke，因为当前工具环境没有可用 CUDA/图形设备，IsaacSim/Kit 报 `No CUDA GPUs are available` / `no suitable CUDA GPU was found` 后退出；这不是 `isaaclab` 模块缺失。
3. MuJoCo sim2sim 真实播放，因为还需要训练出的 policy ONNX/JIT 和完整 MuJoCo 运行依赖。

## 10. 注意事项和排错

### 10.1 reference checkpoint 缺失

如果看到：

```text
BPM reference model checkpoint not found
```

说明没有传入有效 reference checkpoint。使用：

```bash
--reference_model /abs/path/to/reference_state_keypoint_model.pt
```

或修改：

```python
MrobotMimicCfg.motion.reference_model_path
```

### 10.2 IsaacGym gymtorch cache 不可写

如果看到类似：

```text
Read-only file system: .../.cache/torch_extensions/.../gymtorch/lock
```

可尝试使用可写 cache：

```bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
```

如果换 cache 后提示缺少 ninja，需要在该环境安装 ninja，或恢复到已有可用 gymtorch cache。

### 10.3 IsaacLab 缺失

如果运行 IsaacLab 脚本看到：

```text
ModuleNotFoundError: No module named 'isaaclab'
```

说明当前 Python 环境不是 IsaacLab/IsaacSim 环境。切换到安装了 IsaacLab 的环境后再运行 `train_mrobot_isaaclab.py` 或 `play_mrobot_isaaclab.py`。

当前已确认可导入 IsaacLab 的环境包括：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python
/home/weil/anaconda3/envs/issac_sim/bin/python
/home/weil/anaconda3/envs/legged_lab/bin/python
```

如果看到：

```text
No CUDA devices found
no suitable CUDA GPU was found
```

说明 IsaacSim/Kit 没有拿到可用 GPU 或图形/CUDA 驱动。请在你之前能跑通 XBot IsaacLab 的 GPU shell 中执行本文 IsaacLab 命令。

### 10.4 PyTorch 2.6+ checkpoint 加载

如果看到：

```text
Weights only load failed
```

原因是 PyTorch 2.6+ 把 `torch.load()` 默认改成 `weights_only=True`，而 reference checkpoint 内含 numpy metadata。本次已在 reference/env/sim2sim/runner 加载点显式使用 `weights_only=False`。该设置只应对可信本地 checkpoint 使用。

### 10.5 动作维度错误

MRobot policy 输出必须是 12 维。环境内部会扩展到 29 维 full action/DOF。

如果加载旧 policy 时出现 action shape 不匹配，确认：

1. policy 输入是 64 维。
2. policy 输出是 12 维。
3. PPO checkpoint 是用当前 `num_policy_actions=12` 训练的。

### 10.6 AMP 相关报错

本次没有迁移 AMP。不要使用旧 AMP runner、AMP config、AMP checkpoint 直接启动当前任务。

### 10.7 IsaacLab 训练速度慢

如果日志类似：

```text
Computation: 13013 steps/s
collection: 7.445s, learning 0.109s
```

说明慢点主要在 IsaacLab 环境采样/物理仿真，不在 PPO 更新。当前 MRobot 配置为：

1. `num_envs=4096`
2. `num_steps_per_env=24`
3. `sim.dt=0.001`
4. `control.decimation=10`

因此每个 PPO iteration 实际会推进：

```text
4096 * 24 = 98304 个 env control step
98304 * 10 = 983040 个 PhysX step
```

本次已针对 IsaacLab MRobot hot path 做以下优化：

1. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - Contact sensor 当前只监听 termination 相关 body：`base_link / waist_yaw_link / pelvic_yaw_link / knee_pitch_link`。
   - 监听路径为 `/World/envs/env_.*/Robot/.*(base_link|waist_yaw_link|pelvic_yaw_link|knee_pitch_link)`，再按 body name 映射到代码里的 termination body。
   - 原因：全身 contact sensor 在 4096 env 下会明显拖慢 IsaacLab；只监听 termination body 与当前 Gym 终止逻辑一致，并保留 name-based 校验避免顺序漂移。
   - `history_length=1`，`track_air_time=False`，只保留当前帧 net contact force。

2. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_pre_physics_step()` 不再重复调用 `compute_ref_state()`。
   - reference state 在 reset 和 `_get_observations()` 中计算，下一次 policy action 使用的正是上一帧 obs 对应的 reference。
   - 这样每个 env step 的 reference network forward 从 2 次降到 1 次。

3. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_update_state_cache()` 增加 `common_step_counter` 缓存。
   - `_get_dones()` 和 `_get_observations()` 在同一个 DirectRLEnv step 内不再重复刷新 root/body/contact state。

4. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_apply_action()` 复用 `target_dof_pos` 和 `sim_order_torques` buffer，减少每个 physics substep 的临时 tensor 分配。
   - `action_delay_buffer` 从整块 clone 移位改为环形 buffer，保持延迟语义，同时避免每个 physics substep 搬运 `num_envs x 29 x delay` 的大块张量。

复测建议：

```bash
python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 24 \
  --max_iterations 20 \
  --run_name speed_check \
  --no_log
```

如果需要判断 domain randomization 是否是主要慢点，可以对比：

```bash
python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 24 \
  --max_iterations 20 \
  --run_name speed_check_no_dr \
  --no_log \
  --disable_domain_randomization
```

注意：IsaacLab 训练必须加 `--headless` 才能得到合理速度。如果不加 `--headless`，DirectRLEnv 检测到 GUI 后会在 physics loop 中渲染，速度会明显下降。

### 10.8 IsaacGym 与 IsaacLab 训练语义对齐记录

为了解决 “IsaacGym 几十个 iteration 可以站住，但 IsaacLab 需要几百个 iteration” 的问题，已把 Lab 侧进一步向 Gym 侧对齐。核心修改在：

```text
humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py
humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py
```

已对齐内容：

1. Actor yaw 观测
   - Gym：actor obs 中 yaw 使用 `yaw - initial_base_yaw`。
   - Lab：已新增 `initial_base_yaw`，reset 后记录初始 yaw，actor obs 的 euler yaw 改为相对 yaw。

2. Reference 解码
   - Gym：从 reference network 解码 `dof_pos/dof_vel`，以及 `pelvis/feet/knee/hip/pelvic_yaw/waist` 的 `pos/quat/vel/ang_vel`。
   - Lab：已补齐同样 key body reference buffer。
   - Lab reset 时 joint velocity 改为使用 `ref_dof_vel`，不再等价于全部 0 速度。

3. Privileged obs
   - Gym：210 维 privileged obs = 45 维历史/本体信息 + 146 维关键点误差/domain randomization 信息 + 19 维 goal。
   - Lab：已按 Gym 的 146 维结构填入：
     - anchor position/orientation error
     - root lin/ang vel error
     - pelvis/feet/knee/hip/pelvic_yaw/waist 的位置误差
     - pelvis/feet/knee/hip/pelvic_yaw/waist 的 6D 姿态误差
     - push/disturbance/friction/restitution/PD/payload/COM/ref foot contact/phase

4. Whole-body mimic reward
   - Gym：`imitation_whole_body_pos/rot/lin_vel/ang_vel` 基于全身 key bodies。
   - Lab：已由早期简化版改为全 key body 对齐版本，使用与 Gym 相同的 anchor/yaw alignment 语义。

5. Root reward
   - Gym：`imition_root_pos/root_rot` 使用 waist anchor 的局部对齐状态。
   - Lab：已改为使用 waist anchor，而不是只比较 base 高度或 base quaternion。

6. Torque limit reward
   - Gym：只看受控 12 个腿部关节，soft limit 为 `0.9 * torque_limit`，并做归一化和 clamp。
   - Lab：已改成同样的 12 关节归一化形式。

7. DOF position limit reward
   - Gym：计算受控关节越限惩罚。
   - Lab：已从固定返回 0 改为实际计算受控关节越限。

8. Domain randomization curriculum
   - Gym：`OnPolicyRunner` 每个 iteration 调用 env 的 `update_domain_rand_curriculum()`。
   - Lab：`IsaacLabRslRlVecEnv` 已转发 `update_domain_rand_curriculum()` 到 DirectRLEnv。
   - Lab env 已实现 push/disturbance/restitution/payload/COM/link mass/PD/motor strength/motor offset/action delay 的课程式缩放。

仍需注意：

1. IsaacLab 的 URDF -> USD 导入、fixed body merge、contact 求解、材质和 PhysX view 写入时机仍与 IsaacGym 不可能完全一样。
2. Lab 的 contact sensor 现在只监听 termination body，避免全身 contact sensor 成为 4096 env 下的主要瓶颈。
3. contact offset/rest offset randomization 暂未做到与 Gym 完全等价。
4. 如果要判断是否已达到 Gym 训练速度，应先用 `--disable_domain_randomization` 和短训对比，再打开 curriculum。

### 10.9 最新完整对齐补丁记录

本次继续补齐 Lab 侧没有完全迁移的细节，目标是让 `isaaclab_env.py` 的观测、奖励、动作和随机化尽量逐项对应 `mrobot_mimic_env.py`。

修改文件：

```text
humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py
docs/MROBOT_BPM_MIGRATION_PLAN.md
```

`humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py` 新增/修改内容：

1. Contact/termination 对齐
   - contact sensor 改为只采集 termination body，而不是全身 body。
   - 新增 `_validate_tracking_body_indices()`，启动时检查 `base / waist / feet / knee / hip / pelvic_yaw` 的数量是否与 Gym 关键点奖励期望一致。
   - termination 使用 `base_link / waist_yaw_link / pelvic_yaw_link / knee_pitch_link`，并对 contact sensor 的 body name 映射做数量校验。

2. Actor 观测对齐
   - actor obs 保持 `64 = 45 proprio + 19 goal`。
   - yaw 继续使用相对 reset 初始 yaw。
   - 新增 Gym 中已有的脚踝 actor 观测偏置逻辑：
     - `randomize_ankle_obs_pos_bias`
     - `randomize_ankle_obs_vel_bias`
     - `randomize_ankle_obs_vel_noise`
     - `randomize_ankle_obs_vel_delay`
     - `randomize_ankle_obs_vel_filter`
   - 当前默认配置这些脚踝观测污染开关为 False，因此默认训练结果不变；如果以后在 config 中打开，Lab 和 Gym 会走同样的数据流。

3. PD/动作对齐
   - 新增 `_get_ankle_dq_for_pd()`，支持 Gym 里的脚踝 PD 速度反馈 delay/filter/noise：
     - `randomize_ankle_pd_dq_delay`
     - `randomize_ankle_pd_dq_filter`
     - `randomize_ankle_pd_dq_noise`
   - `_apply_action()` 的阻尼项改为使用该速度反馈。
   - 补齐 `randomize_ankle_pd` 和 `randomize_ankle_motor_offset` 的 reset 采样。
   - 保留非受控关节跟随 reference、受控腿部关节使用 reference residual target 的逻辑。

4. sys_delay 对齐
   - 新增 IsaacLab 版 `obs_imu_delay_buffer / obs_motor_delay_buffer`。
   - actor obs 在 `domain_rand.sys_delay=True` 时使用延迟后的 root state 和 joint pos/vel。
   - reset 后会用当前 reference 初始状态填满 delay buffer，避免第一帧 actor obs 读到全 0。
   - 当前默认 `sys_delay=False`，所以默认训练不受影响。

5. Reward/privileged obs 防错
   - whole-body mimic reward 仍使用 `pelvis/base + feet + knee + hip + pelvic_yaw + waist`，anchor 为 `waist_yaw_link`。
   - privileged obs 保持 210 维。
   - `_get_observations()` 末尾增加 actor obs/critic obs 维度硬校验，若改坏会直接报出实际维度和配置维度。

验证记录：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python -m py_compile \
  humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py \
  humanoid_gym_ex/scripts/train_mrobot_isaaclab.py
```

编译检查已通过。

在当前 Codex 执行环境中尝试运行最小 IsaacLab smoke：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --headless \
  --num_envs 2 \
  --num_steps_per_env 2 \
  --max_iterations 1 \
  --no_log \
  --disable_domain_randomization \
  --deterministic_reset \
  --device cuda:0
```

该 smoke 未进入 MRobot env 逻辑，失败原因为当前执行环境没有可见 CUDA：

```text
RuntimeError: No CUDA GPUs are available
```

因此这次只能确认 Python 编译和静态路径；需要在你的本机 `humanoidgym` 环境里用同一条命令复测真实 IsaacLab/PhysX 路径。

### 10.10 IsaacLab 性能与 Gym 语义再对齐

针对 4096 env 训练 collection time 仍偏高、yaw 对齐和 BPM 采样与 Gym 不一致的问题，继续修改：

1. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - contact sensor 的 `update_period` 从 `sim.dt=0.001` 改为 policy step `sim.dt * decimation = 0.01`。
   - termination 仍然每个 control step 检查 `base/waist/pelvic_yaw/knee` 接触，但不再要求 contact sensor 每个 physics substep 更新，降低 IsaacLab contact sensor 开销。

2. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - state cache 使用 IsaacLab `DirectRLEnv.step()` 自带的 `common_step_counter`。
   - 之前 state cache 可能跨 step 复用旧 root/body/contact state；现在每个 DirectRLEnv step 会递增 counter，并强制刷新当前 step 的状态。
   - 后续 10.12 又移除了 MRobot `_get_dones()` 里的重复递增，避免 counter 走两倍速度。

3. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - reference phase 改成 Gym 一样的状态式推进：
     - reset 时设置 `phase_rad = init_phase`
     - 每个 env step 执行 `phase_rad += 2*pi*bpm/60*dt`
   - 不再用 `episode_length_buf` 反推 phase，避免 `init_at_random_ep_len=True` 影响 BPM phase。

4. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - yaw 对齐改成 Gym 版本语义：
     - Gym：`q_diff = cur_anchor_quat * inv(ref_anchor_quat)`，再取 heading quat。
     - Lab：新增 WXYZ 版本 `_calc_heading_quat_wxyz()`，用同样方式计算 heading yaw。

5. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - BPM reset 采样改成 Gym 一致：
     - `sample_integer_bpm=True` 时，`0 BPM` 作为一个候选整数，与 `[bpm_min, bpm_max]` 中每个整数共同采样。
     - 非整数采样且 `include_zero_bpm=True` 时，zero mask 概率按 Gym 版使用 `0.5`。
   - 之前 Lab 版 zero BPM 概率是 `0.05`，与 Gym 不一致。

6. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_get_observations()` 不再重复调用 `compute_ref_state()`。
   - reference 已在每个 step 的 `_get_dones()` 中推进 phase 后计算，reset 时也会计算；这样每个 policy step 的 reference network forward 从 2 次降为 1 次。

7. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_get_rewards()` 不再每个 step 写 `extras["episode"]`。
   - episode info 只在 `_reset_idx()` 中写，避免 PPO runner 每个 iteration 聚合大量非 episode 统计字典。

8. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 删除 Lab 版 `ref_dof_pos` 和 PD `target_dof_pos` 的关节限位 clamp。
   - Gym 版 `_compute_torques()` 与 `sim2sim_mimic.py` 都不对 reference/target 做该 clamp，只通过 torque limit 与 joint limit reward 约束。
   - 如果 Lab 训练时 target 被 clamp，但 MuJoCo sim2sim 中 target 不 clamp，脚踝这类接近限位且经过四连杆映射的关节会最容易出现部署时锯齿抖动。

### 10.11 IsaacLab Collection Time 排查与热路径优化

针对 4096 env、24 step/iteration 时 collection 仍约 3.4s 的问题，继续做不改变训练语义的热路径优化：

1. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 脚踝 actor obs 随机化默认关闭时，`_apply_actor_ankle_obs_bias()` 直接返回原 `q/dq`，不再 clone。
   - 脚踝 PD dq 随机化默认关闭时，`_get_ankle_dq_for_pd()` 直接返回 `self.dof_vel`，不再每个 physics substep clone 29 维速度。

2. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - reference network 输入改为复用预分配 `reference_input` buffer。
   - `compute_ref_state()` 不再每步 `torch.cat()` 生成新输入张量，并使用 `torch.inference_mode()` 包裹 reference MLP forward。

3. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 关键点相对姿态观测从 6 次 `get_rel_pose()` 合并为 1 次对 `all_tracking_indices` 的 vectorized 调用，再按 `pelvis/feet/knee/hip/pelvic_yaw/waist` split。
   - reference 关键点相对姿态也从 6 次 `get_ref_rel_state_current()` 合并为 1 次。

4. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 4 个 whole-body reward 共用 `_get_tracking_reward_cache()`。
   - 当前 body pose、yaw-aligned reference target、body lin/ang vel target 每个 env step 只计算一次，`imitation_whole_body_pos/rot/lin_vel/ang_vel` 不再重复计算 yaw 对齐和 reference cat。

5. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `compute_ref_state()` 之后把 `pelvis/feet/knee/hip/pelvic_yaw/waist` 的 reference pos/quat/lin_vel/ang_vel 写入预分配的 `tracking_ref_*_buf`。
   - reward 和 observation 读取这些缓存，不再在每个 reward/obs 路径里重复 `torch.cat()` reference key body 张量。

6. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_clear_external_forces()` 去掉每个 control step 的 `torch.any(self.external_force_active)` Python 判断。
   - 原写法会在 GPU tensor 上做 `torch.any()` 后进入 Python `if`，这会触发 GPU->CPU 同步；4096 env 下每个 rollout step 都同步一次，会显著拉低 collection FPS。
   - 现在使用 Python bool 标志 `_external_force_active_any` 记录是否真的施加过外力；没有外力时直接返回，不触发 CUDA 同步。

7. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_apply_action()` 只有在 `sys_delay=True` 且 delay buffer 存在时才调用 `_record_sys_delay_state()`。
   - 默认 `sys_delay=False` 时不再每个 physics substep 进入延迟记录函数。

8. `humanoid_gym_ex/algo/ppo/on_policy_runner.py`
   - 新增 `runner.fast_episode_logging` 可选开关。
   - 旧日志路径每个 rollout step 都执行 `nonzero + cpu().numpy()` 记录 done episode，这同样会在训练采样阶段产生 GPU->CPU 同步。
   - 快速路径在 GPU 上累计 done episode 的 reward/length，每个 iteration 结束只同步一次均值，保持训练数据流不变，只改变日志统计的实现方式。

9. `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py`
   - `MrobotMimicLabCfgPPO.runner.fast_episode_logging = True`。
   - 该开关只对 MRobot IsaacLab 训练默认启用，不影响 IsaacGym 训练配置。

10. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
    - 固定 armature 和固定 joint friction 不再每次 reset 都写入 PhysX。
    - 当前默认 `randomize_joint_armature=False`、`randomize_joint_friction=False`，所以这两个属性在初始化时写一次即可；如果以后把对应 randomize 开关打开，仍会在每次 reset 重新采样并写入。
    - 写入 armature/friction 时改为传 `joint_ids=self.joint_sim_ids_list`，避免先构造完整 sim-order action buffer 再切片。

11. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
    - 内部 actor observation 缓存从 `self.obs_buf` 改名为 `self.policy_obs_buf`。
    - 原因：IsaacLab `DirectRLEnv.step()` 会把 `_get_observations()` 的返回值写回 `self.obs_buf`。MRobot `_get_observations()` 返回的是 `{"policy": ..., "critic": ...}` 字典，如果内部也把 `self.obs_buf` 当 tensor 使用，下一步就会出现 `TypeError: unhashable type: 'slice'`。
    - 修改后 `self.obs_buf` 继续留给 IsaacLab 框架管理，MRobot 自己只写 `policy_obs_buf` 和 `privileged_obs_buf`。

这些优化不会改变 obs/reward/action 的数值定义，只减少每步的小 kernel 数量和重复张量分配。若 collection time 仍明显高于预期，下一步应在本机用 IsaacLab 运行时 profiling 区分 PhysX step、contact sensor、reference net、obs/reward 各自耗时。

本次验证命令：

```bash
/home/weil/anaconda3/envs/humanoidgym/bin/python -m py_compile \
  humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py \
  humanoid_gym_ex/algo/ppo/on_policy_runner.py \
  humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py \
  humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py
```

编译检查已通过。由于当前 Codex 运行环境没有可见 CUDA，无法在这里真实跑 IsaacLab 4096 env collection FPS；需要在你的 `humanoidgym` 环境中用原训练命令复测。

### 10.12 IsaacLab 慢速根因排查：高频小 reset 触发 CPU PhysX setter

现象：

- 4096 env、`max_episode_length=1000` 左右时，episode 稳定后平均每个 control step 会有约 `4096 / 1000 ~= 4` 个 env timeout reset。
- PPO 每个 iteration 有 `num_steps_per_env=24`，因此每轮大约有 `4096 * 24 / 1000 ~= 98` 个 env reset。
- `init_at_random_ep_len=True` 会把这些 reset 分散到几乎每个 control step，而不是集中在某一个 iteration。

真正拖慢的路径：

1. IsaacLab `DirectRLEnv.step()` 每个 control step 都会执行：
   - `_get_dones()`
   - 找出 `reset_env_ids`
   - 对这些 env 调 `_reset_idx(reset_env_ids)`
2. MRobot Lab 之前在 `_reset_idx()` -> `_randomize_reset_buffers()` 中，每次小 reset 都调用：
   - `_randomize_materials()` -> `root_physx_view.get_material_properties()` / `set_material_properties()`
   - `_randomize_mass_and_com()` -> `set_masses()` / `set_inertias()` / `set_coms()`
   - `_randomize_joint_physx_props()` -> `write_joint_armature_to_sim()` / `write_joint_friction_coefficient_to_sim()`
3. 这些 IsaacLab / PhysX API 都是 CPU tensor setter 路径。IsaacLab 官方事件实现中 mass/COM randomization 也明确使用 `env_ids.cpu()`、`get_masses()`、`get_inertias()`、`get_coms()`，并提示 COM 这类写入更适合 initialization event。

历史修改：

1. `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py`
   - 新增：
     ```python
     resample_physx_randomization_on_small_reset = False
     ```
   - 默认含义：material / mass / inertia / COM / joint PhysX 属性只在全环境 reset 和 curriculum stage change 时重采样并写入 PhysX。
   - 小批量 episode reset 时不再重采样这些 CPU PhysX 属性，避免 rollout 中几乎每步都走 CPU setter。

2. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 新增 `_should_resample_physx_randomization(env_ids)`。
   - `_randomize_reset_buffers()` 中只有该函数返回 True 时才调用：
     - `_randomize_materials(env_ids)`
     - `_randomize_mass_and_com(env_ids)`
     - `_randomize_joint_physx_props(env_ids)`
   - GPU 上便宜的随机化仍然每次 reset 生效：
     - `randomize_kp`
     - `randomize_kd`
     - `randomize_motor_strength`
     - `randomize_motor_offset`
     - `randomize_default_dof_pos_offset`
     - `action_delay`
     - ankle obs / ankle PD 相关随机化

3. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 修复 `common_step_counter` 双倍递增。
   - IsaacLab `DirectRLEnv.step()` 已经在每个 env step 后执行 `self.common_step_counter += 1`；MRobot `_get_dones()` 里不应该再次递增。
   - 之前双倍递增会让 push 间隔实际变成配置的一半，例如 `[1.0, 3.0]s` 近似变成 `[0.5, 1.5]s`。

4. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - adaptive curriculum 的 episode length / fall ratio 不再在每个 reset 中 `.item()`。
   - reset 时只在 GPU 上累计 sum/count；`update_domain_rand_curriculum()` 每个 PPO iteration 开头统一同步一次。

为什么这更接近根因：

- 之前的优化主要减少 obs/reward 小 kernel 和日志同步。
- 本次修改切掉的是 rollout 过程中反复发生的 CPU PhysX property 写入；这类 API 会把小批量 reset 放大成全局 CPU/PhysX 操作，是 4096 env IsaacLab 下最符合 `collection time` 长期偏高的路径。
- 如果这次修改后 collection time 明显下降，说明根因就是小 reset 高频写 PhysX；如果仍高，则下一步应打开更细粒度 profiling，继续分解 PhysX step / scene.update / obs / reward / reset 的耗时。

### 10.13 重新对齐 Gym：每个 episode reset 重采样 Lab 物理随机化

用户要求 IsaacLab 和 IsaacGym 的 domain randomization 语义继续对齐，因此当前代码已经把上一节的性能折中撤回：

1. `humanoid_gym_ex/envs/robots/mrobot/mrobot_mimic_config_lab.py`
   - 当前值：
     ```python
     resample_physx_randomization_on_small_reset = True
     ```
   - 含义：每个 env 的 episode reset 都重新采样并尝试写入：
     - material friction / restitution
     - link mass
     - waist payload mass
     - waist COM displacement
     - joint armature / joint friction PhysX 属性（仅在对应随机化开启时每次重采样；当前 armature 是固定值，首次写入后保持不变）

2. 同一文件中关闭 Lab 的脚踝专属随机化：
   - `randomize_ankle_pd = False`
   - `randomize_ankle_motor_offset = False`
   - `randomize_ankle_obs_pos_bias = False`
   - `randomize_ankle_obs_vel_bias = False`
   - `randomize_ankle_obs_vel_noise = False`
   - `randomize_ankle_obs_vel_delay = False`
   - `randomize_ankle_obs_vel_filter = False`
   - `randomize_ankle_pd_dq_noise = False`
   - `randomize_ankle_pd_dq_delay = False`
   - `randomize_ankle_pd_dq_filter = False`
   - `default_dof_pos_offset_ankle_range = None`

普通全关节随机化仍然保留，例如 `randomize_kp`、`randomize_kd`、`randomize_motor_strength`、`randomize_default_dof_pos_offset` 等。`ankle_dof_vel`、`ankle_torque_limit` 是奖励项，不属于脚踝随机化，没有关闭。

### 10.14 IsaacLab step profiling 开关

为了继续定位 Lab collection time，新增了不改变训练语义的 profiling 开关：

1. `humanoid_gym_ex/scripts/train_mrobot_isaaclab.py`
   - 新增参数：
     ```bash
     --profile_step_timings
     --profile_step_timing_interval 200
     --profile_step_timing_warmup 20
     ```

2. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 在 `MrobotMimicIsaacLabEnv.step()` 中，仅当 `profile_step_timings=True` 时覆盖 DirectRLEnv 的默认 step，并用 CUDA synchronize + wall time 分段统计：
     - `pre_physics_step`
     - `apply_action`
     - `write_data_to_sim`
     - `sim_step`
     - `render`
     - `scene_update`
     - `dones`
     - `rewards`
     - `reset`
     - `observations`
     - `total`

使用示例：

```bash
conda run -n humanoidgym python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --num_steps_per_env 24 \
  --max_iterations 5 \
  --profile_step_timings \
  --profile_step_timing_warmup 20 \
  --profile_step_timing_interval 100
```

注意：profiling 会强制 CUDA 同步，本身会让训练变慢。它只用于定位耗时，正式训练不要打开。

### 10.15 根据 profiling 继续优化 reset 路径

实测 profile：

```text
pre_physics_step=0.180ms
apply_action=3.135ms
write_data_to_sim=2.869ms
sim_step=40.084ms
scene_update=2.372ms
dones=22.403ms
rewards=2.116ms
reset=44.555ms
observations=4.615ms
total=122.406ms
```

结论：

- PPO learning 不是瓶颈。
- `sim_step` 是 IsaacLab / PhysX 主物理步进成本。
- `reset` 主要来自每 episode 重采样 PhysX 属性时的 CPU setter。
- `dones` 需要继续拆分 reference network、state cache、push/contact/termination。

已修改：

1. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - 对 material / mass / inertia / COM 的 PhysX CPU buffer 做缓存：
     - `_physx_materials_cpu`
     - `_physx_masses_cpu`
     - `_physx_inertias_cpu`
     - `_physx_coms_cpu`
   - 每次 episode reset 仍然重新采样，但只更新对应 `env_ids` 的 CPU buffer 行，不再反复 `get_material_properties()` 或 clone 全量默认 mass/inertia/com。
   - 写回 PhysX 的 setter 仍然保留，因此 Gym 语义不变。

2. 同一文件中，PhysX 属性随机化改为 CPU 采样：
   - 原逻辑：GPU 采样 -> `.detach().cpu()` -> PhysX setter。
   - 新逻辑：CPU 采样 -> PhysX setter，同时把少量随机化结果同步回 GPU buffer 供 privileged obs 使用。
   - 这样避免 GPU 到 CPU 的隐式同步。

3. profiling 输出新增细分字段：
   - `dones_phase_ref`
   - `dones_state_cache`
   - `dones_push_contact`
   - `dones_termination`
   - `reset_episode_logging`
   - `reset_robot_super`
   - `reset_bpm_ref`
   - `reset_domain_rand`
   - `reset_state_write`
   - `reset_cleanup`

复测命令不变：

```bash
conda run -n humanoidgym python humanoid_gym_ex/scripts/train_mrobot_isaaclab.py \
  --task mrobot_music \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --num_steps_per_env 24 \
  --max_iterations 5 \
  --profile_step_timings \
  --profile_step_timing_warmup 20 \
  --profile_step_timing_interval 100
```

### 10.16 第二轮 profiling：state cache 是新的瓶颈

第二轮实测：

```text
dones=21.964ms
dones_phase_ref=2.746ms
dones_state_cache=18.764ms
reset=24.005ms
reset_domain_rand=3.196ms
reset_cleanup=15.378ms
total=100.609ms
```

结论：

- 上一轮 reset 优化有效，`reset` 从约 `44.6ms` 降到约 `24.0ms`。
- 当前最大的可疑路径是 `_update_state_cache()`：
  - `dones_state_cache`
  - `reset_cleanup`
- 由于 early training 中 `fall_ratio=1.0`，平均 episode length 约 200，4096 env 会导致几乎每个 control step 都有二十多个 env reset。reset 频率本身仍然很高。

已修改：

1. `humanoid_gym_ex/envs/robots/mrobot/isaaclab_env.py`
   - `_update_state_cache()` 新增细分 profile：
     - `state_root_joint`
     - `state_base_vel`
     - `state_body`
   - termination 的 contact force 读取从 `_update_state_cache()` 中移出，单独计入：
     - `dones_contact_read`

2. 同一文件中，base body-frame 速度不再读取 IsaacLab lazy property：
   - 原逻辑：
     ```python
     self.base_lin_vel = self.robot.data.root_lin_vel_b
     self.base_ang_vel = self.robot.data.root_ang_vel_b
     ```
   - 新逻辑：
     ```python
     self.base_lin_vel = _quat_rotate_inverse_wxyz(self.base_quat, self.root_states[:, 7:10])
     self.base_ang_vel = _quat_rotate_inverse_wxyz(self.base_quat, self.root_states[:, 10:13])
     ```
   - 数值语义保持一致，但避免 IsaacLab data object 的 lazy property 额外开销。

3. 同一文件中，腰部 anchor body id 在初始化时缓存：
  - `self.waist_body_id`
  - `self.base_body_id`
  - 避免在 reward/obs 的热路径中反复对 GPU tensor 调 `.item()`。

### 10.17 record_bpm_keypoints.py 改为 IsaacLab/IsaacSim 路径

文件：

- `humanoid_gym_ex/scripts/record_bpm_keypoints.py`

修改目标：

- 原脚本通过 IsaacGym `gymapi/gymtorch` 创建 `mrobot_music` 环境，逐帧写入 root / DOF state 后读取 `rigid_body_state_tensor`。
- 当前已改为通过 IsaacLab/IsaacSim 跑一遍数据，逐帧写入 `Articulation` root / joint state，并从 `robot.data.body_state_w` 获取关键点信息。

主要代码变化：

1. 删除 IsaacGym 依赖：
   - 移除 `from isaacgym import gymapi, gymtorch`
   - 移除 `humanoid.utils.get_args`
   - 移除 `task_registry.make_env`
   - 移除 `env.gym.set_dof_state_tensor_indexed`
   - 移除 `env.gym.refresh_rigid_body_state_tensor`

2. 新增 IsaacLab 启动：
   - 使用 `isaaclab.app.AppLauncher`
   - 支持 IsaacLab 通用参数：
     - `--headless`
     - `--device cuda:0`
     - `--rendering_mode`
     - `--kit_args`

3. 环境创建：
   - 使用 `MrobotMimicIsaacLabEnvCfg`
   - 使用 `MrobotMimicIsaacLabEnv`
   - 固定 `num_envs=1`
   - `disable_domain_randomization=True`
   - `deterministic_reset=True`
   - 可选 `--reference_model` 覆盖 reference checkpoint 路径

4. 每帧写入方式：
   ```python
   env.robot.write_root_pose_to_sim(root_states[:, :7], env_ids)
   env.robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)
   env.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
   env.scene.write_data_to_sim()
   env.sim.forward()
   env.scene.update(dt=env.physics_dt)
   ```

5. 每帧读取方式：
   ```python
   rigid_state = env.robot.data.body_state_w[:, indices]
   ```

6. 四元数顺序：
   - 输入 base quat：`w,x,y,z`
   - IsaacLab body_state：`w,x,y,z`
   - 输出 keypoint quat：转换成 `x,y,z,w`
   - 原因：reference network 数据读取逻辑按列名 `*_quat_*_x/y/z/w` 读取，然后在 env 内部转换成 `w,x,y,z`；为了兼容旧数据格式，CSV 输出仍保持 `x,y,z,w`。

常用命令：

```bash
conda run -n humanoidgym python humanoid_gym_ex/scripts/record_bpm_keypoints.py \
  --headless \
  --device cuda:0 \
  --input_dir BPM_dance/bpm_phase_state_dataset \
  --output_dir BPM_dance/bpm_phase_state_dataset_keypoint \
  --bpm_start 60 \
  --bpm_end 170
```

单文件检查：

```bash
conda run -n humanoidgym python humanoid_gym_ex/scripts/record_bpm_keypoints.py \
  --headless \
  --device cuda:0 \
  --input_dir BPM_dance/bpm_phase_state_dataset \
  --output_dir BPM_dance/bpm_phase_state_dataset_keypoint \
  --file bpm_150.csv
```

打开 IsaacSim 画面检查：

```bash
conda run -n humanoidgym python humanoid_gym_ex/scripts/record_bpm_keypoints.py \
  --device cuda:0 \
  --input_dir BPM_dance/bpm_phase_state_dataset \
  --file bpm_150.csv \
  --render \
  --no_save
```

验证：

```bash
python -m py_compile humanoid_gym_ex/scripts/record_bpm_keypoints.py
conda run -n humanoidgym python humanoid_gym_ex/scripts/record_bpm_keypoints.py --help
```

以上两项已通过。当前 Codex 会话没有可用 CUDA，因此没有在这里实际启动 IsaacSim 逐帧生成 CSV；需要在你的 `humanoidgym` 环境和 GPU 可用的终端运行上面的命令。

## 11. 后续建议

1. 用真实 reference checkpoint 跑 IsaacGym `--num_envs 16 --max_iterations 1` smoke。
2. smoke 通过后再跑 4096 env 正式训练。
3. 用训练出的 policy 导出 TorchScript/ONNX 后再跑 `sim2sim_mimic.py`。
4. IsaacLab 版本先用小 env 数确认 reset/step/obs/action/reward shape，再逐步补齐与 IsaacGym 的 domain randomization 等价性。
