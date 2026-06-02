# import torch
#  4.23 新的改了關鍵點的


# # 1. 加载模型
# model_path = "/home/weil/hl_rl/hl_rl/logs/mrobot_mimic/exported/policies/policy_JT.pt"
# model = torch.jit.load(model_path)
# model.eval()

# # 2. 输入维度：15帧历史 + goal = 45*15 + 46 = 721
# # 模型内部会自动切分处理，你不需要管
# input_dim = 721
# example_input = torch.randn(1, input_dim)

# print(f"正在导出模型，输入维度: {example_input.shape} ...")

# # 3. 导出 ONNX
# torch.onnx.export(
#     model,
#     example_input,
#     "casbot_end2end.onnx", # 输出文件名
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=['obs'],     # 部署时，这个 obs 就是那个 721 维的大数组
#     output_names=['actions']
# )

# print("导出成功！")

'''
import torch
import torch.nn as nn
import copy
import os

# ==========================================
# 1. 适配 15 帧 (821维) 的组装模型
# ==========================================
class ReAssembledPolicy(nn.Module):
    def __init__(self, jit_model):
        super().__init__()
        # 提取子模块
        try:
            self.encoder = copy.deepcopy(jit_model.history_encoder)
            self.actor = copy.deepcopy(jit_model.actor)
        except Exception:
            self.encoder = jit_model.history_encoder
            self.actor = jit_model.actor

    def forward(self, full_obs):
        # --- 输入定义 ---
        # full_obs 形状: [1, 821]  <-- 修正为 15 帧的总维度
        
        # 1. 提取历史 (前 14 帧)
        # 索引: 0 ~ 630
        history = full_obs[:, :630]
        
        # 2. 提取当前观测 (第 15 帧)
        # 索引: 630 ~ 675
        obs_now = full_obs[:, 630:675]
        
        # 3. 提取目标信息
        # 索引: 675 ~ 821 (675 + 146 = 821)
        goal = full_obs[:, 675:]
        
        # --- 执行计算 ---
        # 编码历史 (630 -> 64)
        latent = self.encoder(history)
        
        # 拼接 (45 + 64 + 146 = 255)
        # 顺序: [obs_now, latent, goal]
        actor_input = torch.cat([obs_now, latent, goal], dim=1)
        
        # 执行 Actor
        actions = self.actor(actor_input)
        
        return actions

# ==========================================
# 2. 导出函数
# ==========================================
def export_onnx():
    model_path = "/home/weil/hl_rl/hl_rl/logs/mrobot_mimic/exported/policies/policy_JT.pt" 
    output_path = "casbot_15frames_821.onnx"

    print(f"1. 加载模型: {model_path}")
    if not os.path.exists(model_path):
        print("错误: 找不到模型文件")
        return
        
    jit_model = torch.jit.load(model_path)
    jit_model.eval()

    print("2. 组装模型 (821维输入)...")
    deploy_model = ReAssembledPolicy(jit_model)
    deploy_model.eval()
    
    # 编译为 JIT 防止 trace 报错
    try:
        deploy_model = torch.jit.script(deploy_model)
    except Exception as e:
        print(f"编译警告: {e}")

    # --- 关键修改：输入维度改为 821 ---
    dummy_input = torch.randn(1, 821)
    
    print(f"3. 开始导出 (Input: 821 -> Internal: 255 -> Action)...")
    
    torch.onnx.export(
        deploy_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['actions']
    )
    print(f"\n✅ 导出成功！保存为: {output_path}")

if __name__ == "__main__":
    export_onnx()
'''
import torch
import torch.nn as nn
import copy
import os
import sys
import argparse
import importlib.util
import numpy as np

# 从 checkpoint 导出时用的配置（与 humanoid/envs/custom/mrobot_mimic_config.py 保持一致，避免 import humanoid.envs 触发 isaacgym）
MRobot_MIMIC_CONFIG = {
    "num_observations": 45 + 19,
    "num_privileged_obs": 45 + 146 + 19,
    "num_control": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "num_aux": 9,  # 65  9
    "num_single_obs": 45,
    "num_goal_obs": 19,
    "actor_hidden_dims": [512, 256, 128],
     "critic_hidden_dims": [512, 256, 128],
    # "critic_hidden_dims": [768, 256, 128],
}

# 观测维度
NUM_OBS = MRobot_MIMIC_CONFIG["num_observations"]


# ==========================================
# 0. 归一化模块（与 EmpiricalNormalization 的 mean/std 一致，便于从 checkpoint 加载）
# ==========================================
class NormalizerModule(nn.Module):
    """仅做 (x - mean) / (std + eps)，用于导出 ONNX 时把归一化 baked in。"""

    def __init__(self, shape, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, shape))
        self.register_buffer("_std", torch.ones(1, shape))

    def load_from_empirical_state_dict(self, state_dict):
        """从 runner 保存的 obs_normalizer.state_dict() 加载 _mean, _std。"""
        self._mean.copy_(state_dict["_mean"])
        self._std.copy_(state_dict["_std"])

    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)


# ==========================================
# 1. 定义适配维度的端到端模型 (Wrapper)
# ==========================================
class ReAssembledPolicy(nn.Module):
    """输入完整 obs -> 输出动作。"""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    @classmethod
    def from_actor_critic(cls, actor_critic):
        return cls(actor=actor_critic.actor.cpu())

    def forward(self, full_obs):
        return self.actor(full_obs)


class PolicyWithNormalizer(nn.Module):
    """先对观测做归一化再送进 policy，导出 ONNX 时归一化被 baked in，部署端只需送 raw obs。"""

    def __init__(self, normalizer: NormalizerModule, policy: nn.Module):
        super().__init__()
        self.normalizer = normalizer
        self.policy = policy

    def forward(self, obs):
        return self.policy(self.normalizer(obs))


# ==========================================
# 2. 仅用训练 checkpoint 一步导出 ONNX（推荐：无需先跑 play 导 JIT）
# ==========================================
def export_onnx_from_ckpt(ckpt_path, output_path):
    """
    从训练保存的 model_XXX.pt 直接导出 ONNX，自动带归一化（若 checkpoint 中有 obs_normalizer）。
    只需一个路径，无需先导出 JIT。
    """
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到 checkpoint: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        print("错误: checkpoint 中无 model_state_dict，请使用训练保存的 .pt 文件")
        return

    cfg = MRobot_MIMIC_CONFIG
    num_obs = cfg["num_observations"]
    num_privileged_obs = cfg["num_privileged_obs"]
    num_actions = len(cfg["num_control"])
    num_aux = cfg["num_aux"]
    # 只加载 actor_critic.py，不经过 humanoid.algo.ppo.__init__（避免牵出 isaacgym）
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _actor_critic_path = os.path.join(_root, "algo", "ppo", "actor_critic.py")
    spec = importlib.util.spec_from_file_location("actor_critic", _actor_critic_path)
    _ac_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ac_mod)
    ActorCritic = _ac_mod.ActorCritic

    actor_critic = ActorCritic(
        num_obs,
        num_privileged_obs,
        num_actions,
        num_aux,
        actor_hidden_dims=cfg["actor_hidden_dims"],
        critic_hidden_dims=cfg["critic_hidden_dims"],
        init_noise_std=np.ones(num_actions, dtype=np.float32),
    ).cpu()
    actor_critic.load_state_dict(ckpt["model_state_dict"], strict=False)
    actor_critic.eval()

    deploy_model = ReAssembledPolicy.from_actor_critic(actor_critic)
    deploy_model.eval()

    if "obs_normalizer" in ckpt:
        print("从 checkpoint 加载观测归一化，ONNX 将包含归一化 (输入为 raw obs)")
        normalizer = NormalizerModule(shape=NUM_OBS)
        normalizer.load_from_empirical_state_dict(ckpt["obs_normalizer"])
        deploy_model = PolicyWithNormalizer(normalizer, deploy_model)
        deploy_model.eval()
    else:
        print("checkpoint 中无 obs_normalizer，导出不含归一化")

    dummy_input = torch.randn(1, NUM_OBS, device="cpu")
    try:
        with torch.no_grad():
            test_out = deploy_model(dummy_input)
        print(f"前向测试通过，输出维度: {test_out.shape}")
    except Exception as e:
        print(f"前向测试失败: {e}")
        return

    torch.onnx.export(
        deploy_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
    )
    print(f"SUCCESS! 仅用 checkpoint 一步导出: {output_path}")


# ==========================================
# 3. 从 JIT 导出（兼容旧流程，可选 ckpt 只做归一化）
# ==========================================
def export_onnx(jit_path, output_path, ckpt_path=None):
    """
    jit_path: 已导出的 JIT policy (encoder+actor)
    output_path: 输出的 ONNX 路径
    ckpt_path: 可选，训练 checkpoint (.pt)，若含有 obs_normalizer 则把归一化 baked 进 ONNX
    """
    print(f"正在加载 JIT 模型: {jit_path}")
    if not os.path.exists(jit_path):
        print("错误: 找不到 JIT 模型文件")
        return
    jit_model = torch.jit.load(jit_path, map_location='cpu')
    jit_model.eval()

    deploy_model = ReAssembledPolicy(jit_model)
    deploy_model.eval()

    print("正在编译模型逻辑...")
    deploy_model = torch.jit.script(deploy_model)

    # 若提供 checkpoint 且含 normalizer，则包一层归一化再导出
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if "obs_normalizer" in ckpt:
            print(f"从 checkpoint 加载观测归一化参数: {ckpt_path}")
            normalizer = NormalizerModule(shape=NUM_OBS)
            normalizer.load_from_empirical_state_dict(ckpt["obs_normalizer"])
            deploy_model = PolicyWithNormalizer(normalizer, deploy_model)
            deploy_model.eval()
            print("ONNX 将包含归一化 (输入为 raw obs)")
        else:
            print("checkpoint 中无 obs_normalizer，导出不含归一化")
    else:
        if ckpt_path:
            print(f"未找到 checkpoint 或未指定: {ckpt_path}，导出不含归一化")

    dummy_input = torch.randn(1, NUM_OBS, device='cpu')
    print(f"开始导出 ONNX (输入维度: {NUM_OBS})...")

    try:
        with torch.no_grad():
            test_out = deploy_model(dummy_input)
        print(f"前向传播测试通过！输出动作维度: {test_out.shape}")
    except Exception as e:
        print(f"❌ 错误: 模型维度不匹配\n详情: {e}")
        return

    torch.onnx.export(
        deploy_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['actions']
    )
    print(f"SUCCESS! 导出成功: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导出 mrobot_mimic policy 为 ONNX。推荐：仅传 --ckpt_path 一步完成（含归一化）；也可传 --jit_path 走旧流程。"
    )
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="训练 checkpoint (model_XXX.pt)：仅此一个参数即可导出 ONNX，自动带归一化")
    parser.add_argument("--jit_path", type=str, default=None,
                        help="可选，JIT policy 路径；不传则从 --ckpt_path 直接构建策略")
    parser.add_argument("--output", "-o", type=str,
                        default=os.path.join(os.path.dirname(__file__), "casbot_mimic.onnx"),
                        help="输出 ONNX 路径")
    args = parser.parse_args()

    if args.ckpt_path and not args.jit_path:
        # 推荐：只给训练 .pt，一步导出 ONNX（无需先 play 导 JIT）
        export_onnx_from_ckpt(args.ckpt_path, args.output)
    elif args.jit_path:
        export_onnx(args.jit_path, args.output, args.ckpt_path)
    else:
        print("请至少指定 --ckpt_path（推荐）或 --jit_path。")
        print("示例: python deploy/export_actor.py --ckpt_path logs/mrobot_mimic/xxx/model_1000.pt -o deploy/casbot.onnx")

    # python humanoid_gym_ex/deploy/export_actor.py --ckpt_path logs/mrobot_mimic_May_music_BPM_isaaclab/May29_15-05-46_lab/model_6750.pt -o humanoid_gym_ex/deploy/casbot_bpm_lab_0529_2.onnx
    # python humanoid/scripts/sim2sim.py --load_model deploy/casbot.onnx
