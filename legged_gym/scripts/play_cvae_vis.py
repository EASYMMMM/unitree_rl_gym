# legged_gym/scripts/play_cvae_vis.py

import sys
import os
import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR

import numpy as np
import torch
import time
import matplotlib.pyplot as plt

# 引入 sklearn 和 scipy 进行科学计算
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

# ----------------------------------------------------------------------------
# 验证函数集 (使用 sklearn/scipy)
# ----------------------------------------------------------------------------

def collect_verification_data(env, policy, num_steps=300, device='cuda'):
    """
    运行环境和策略，收集 Latent z 和 真实动力学参数 epsilon。
    """
    print(f"Collecting data for {num_steps} steps...")
    policy.eval()
    
    # [FIX] env.reset() 返回 (obs, privileged_obs)，我们需要解包
    obs, _ = env.reset()
    
    # Data containers
    z_history = []       # [steps, num_envs, z_dim]
    epsilon_history = [] # [steps, num_envs, eps_dim]
    
    with torch.no_grad():
        for i in range(num_steps):
            # 1. Get Latent z from Policy Encoder
            # Encoder returns: vt, z, mu_z, logstd_z
            # 我们使用 mu_z (确定性部分) 进行分析
            _, _, mu_z, _ = policy.encoder(obs)
            
            # 2. Get Ground Truth Epsilon (Dynamics Params)
            if hasattr(env, 'dynamics_params_buf') and env.dynamics_params_buf is not None:
                epsilon = env.dynamics_params_buf.clone()
            else:
                epsilon = torch.zeros(env.num_envs, 4, device=device) 
            
            # [DEBUG] Print epsilon at the first step
            if i == 0:
                print("\n" + "-"*40)
                print(f"[DEBUG] Checking Epsilon (Dynamics Params) at Step {i}:")
                print(f"  Shape: {epsilon.shape}")
                print(f"  First 15 Envs (Sample):\n{epsilon[:15].cpu().numpy()}")
                print(f"  Mean across envs: {epsilon.mean(dim=0).cpu().numpy()}")
                print(f"  Std  across envs: {epsilon.std(dim=0).cpu().numpy()}")
                
                is_all_zero = torch.all(epsilon == 0)
                is_constant = torch.max(epsilon.std(dim=0)) < 1e-6
                
                if is_all_zero:
                    print("  [WARNING] Epsilon IS ALL ZEROS! Environment randomization might be OFF or broken.")
                elif is_constant:
                    print("  [WARNING] Epsilon variance is near zero! All envs have same parameters.")
                else:
                    print("  [OK] Epsilon looks randomized.")
                print("-"*40 + "\n")

            # Record
            z_history.append(mu_z.cpu())
            epsilon_history.append(epsilon.cpu())
            
            # 3. Step Environment
            # act_inference 需要 tensor 类型的 obs
            actions = policy.act_inference(obs)
            
            # [FIX] env.step 返回 (obs, privileged_obs, rews, dones, infos)
            # 这里的 obs 会被更新为 Tensor，用于下一次循环
            obs, _, _, _, infos = env.step(actions)
            
            # 如果 step 后 info 里有 dynamic_params，用它覆盖 (更准确)
            if 'dynamic_params' in infos:
                epsilon_history[-1] = infos['dynamic_params'].cpu()

    # Stack into tensors: [Time, Batch, Dim]
    z_tensor = torch.stack(z_history, dim=0) 
    eps_tensor = torch.stack(epsilon_history, dim=0)
    
    return z_tensor, eps_tensor

def analyze_consistency(z_tensor):
    """
    验证目标 1: 单个 episode 内，z 是否处在一片小区域内。
    """
    # z_tensor: [Time, Batch, Z_Dim]
    std_over_time = torch.std(z_tensor, dim=0) # 对时间维度求标准差
    avg_std = torch.mean(std_over_time).item() # 对 Batch 和 Z维度求平均
    
    # 计算全局标准差作为对比基准
    global_std = torch.std(z_tensor.reshape(-1, z_tensor.shape[-1]), dim=0).mean().item()
    
    ratio = avg_std / (global_std + 1e-6)
    
    print(f"\n[Consistency Analysis]")
    print(f"  > Intra-episode Std (Stability): {avg_std:.4f} (越低越好)")
    print(f"  > Global Std (Diversity):        {global_std:.4f}")
    print(f"  > Noise-to-Signal Ratio:         {ratio:.2%} (应 < 10% 为佳)")
    
    return avg_std, ratio

def analyze_proximity_and_structure(z_tensor, eps_tensor):
    """
    验证目标 2: 相近的 epsilon 应该能得到相近的 z。
    使用 scipy 计算距离相关性。
    """
    # 取时间平均，得到每个 episode 的代表性 z 和 epsilon
    z_mean = torch.mean(z_tensor, dim=0).numpy()     # [Batch, Z_Dim]
    eps_mean = torch.mean(eps_tensor, dim=0).numpy() # [Batch, Eps_Dim]
    
    # 1. 计算成对距离矩阵 (condensed form)
    dist_eps = pdist(eps_mean, metric='euclidean')
    dist_z = pdist(z_mean, metric='euclidean')
    
    # 2. 计算皮尔逊相关系数
    if len(dist_eps) > 0:
        correlation, _ = pearsonr(dist_eps, dist_z)
    else:
        correlation = 0.0
    
    print(f"\n[Proximity & Structure Analysis]")
    print(f"  > Distance Correlation (rho):    {correlation:.4f} (接近 1.0 表示结构对齐极好)")
    print(f"    (rho > 0.5 通常意味着学到了显著的物理流形)")

    return z_mean, eps_mean, correlation

def visualize_latent_space(z_mean, eps_mean):
    """
    可视化: 使用 sklearn PCA 投影并按物理参数染色
    """
    batch_size, eps_dim = eps_mean.shape
    
    # PCA 降维到 2D
    reducer = PCA(n_components=2)
    z_2d = reducer.fit_transform(z_mean)
    
    # 参数名称映射 (根据 g1_m_env.py 的实现推断)
    # dynamics_params_buf 顺序通常是: [friction, mass, motor_f, motor_d]
    param_names = ['Friction', 'Mass', 'Motor Friction', 'Motor Damping']
    
    # 绘图
    num_plots = min(eps_dim, 4)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1: axes = [axes]
    
    for i in range(num_plots): 
        ax = axes[i]
        param_values = eps_mean[:, i]
        
        # 绘制散点图
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=param_values, cmap='viridis', alpha=0.8, s=30)
        
        title = param_names[i] if i < len(param_names) else f'Param {i}'
        ax.set_title(f"Colored by {title}")
        ax.set_xlabel("PC 1")
        if i == 0: ax.set_ylabel("PC 2")
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    plt.show()

def run_verification(env, policy, device='cuda'):
    print("Starting Z-Space Verification (Sklearn Version)...")
    
    # 1. Collect
    z_tensor, eps_tensor = collect_verification_data(env, policy, device=device)
    
    # 2. Consistency
    analyze_consistency(z_tensor)
    
    # 3. Proximity
    z_mean, eps_mean, corr = analyze_proximity_and_structure(z_tensor, eps_tensor)
    
    # 4. Visualization
    try:
        visualize_latent_space(z_mean, eps_mean)
        print("\nVisualization generated. Check plots.")
    except Exception as e:
        print(f"\nVisualization failed: {e}")

# ----------------------------------------------------------------------------
# 主 Play 函数
# ----------------------------------------------------------------------------

def play(args):
    # 1. 获取并覆盖配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 覆盖环境配置
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100) 
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    
    # 强制开启相关参数以匹配训练时的随机化设置，确保验证的有效性
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False 

    # 准备 Runner 配置
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run 
    
    # [FIX] 设置正确的日志根目录
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    
    # [FIX] 显式计算并设置模型路径，处理 checkpoint 参数
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    train_cfg.runner.resume_path = get_load_path(root=log_root, 
                                                 load_run=args.load_run, 
                                                 checkpoint=checkpoint)

    # 2. 创建环境和 Runner
    print(f"Loading environment: {args.task}")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    print(f"Loading run: {args.load_run}")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # 3. 加载模型权重
    print(f"Loading model from: {train_cfg.runner.resume_path}")
    ppo_runner.load(train_cfg.runner.resume_path)
    
    policy = ppo_runner.alg.actor_critic
    policy.eval() 
    
    print("\n" + "="*60)
    print(" DPCVAE Z-Space Verification Phase")
    print("="*60 + "\n")
    
    # 4. 执行 Z 空间验证逻辑
    run_verification(env, policy, device=env.device)

    print("\n" + "="*60)
    print(" Visualization Loop Starting... (Close window to exit)")
    print("="*60 + "\n")

    # 5. 进入常规可视化循环
    # [FIX] 这里同样需要解包 reset 返回的元组
    obs, _ = env.reset()
    
    # if args.record_video:
    #     video_dir = os.path.join(log_root, 'videos')
    #     os.makedirs(video_dir, exist_ok=True)
    #     print(f"Recording video to {video_dir}...")

    t_start = time.time()
    step_cnt = 0
    
    try:
        while True:
            with torch.no_grad():
                actions = policy.act_inference(obs)
            
            obs, _, rews, dones, infos = env.step(actions)
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                print(f"Simulating step {step_cnt}...")

    except KeyboardInterrupt:
        print("EXITING.")

if __name__ == '__main__':
    args = get_args()
    play(args)