import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def collect_verification_data(env, policy, num_steps=300, device='cuda'):
    """
    运行环境和策略，收集 Latent z 和 真实动力学参数 epsilon。
    """
    print(f"Collecting data for {num_steps} steps...")
    policy.eval()
    
    # Reset Environment
    obs = env.reset()
    
    # Data containers
    z_history = []       # [steps, num_envs, z_dim]
    epsilon_history = [] # [steps, num_envs, eps_dim]
    
    with torch.no_grad():
        for i in range(num_steps):
            # 1. Get Latent z from Policy Encoder
            # Encoder returns: vt, z, mu_z, logstd_z
            # 我们通常使用均值 mu_z 进行分析，因为它代表确定性的编码中心
            _, _, mu_z, _ = policy.encoder(obs)
            
            # 2. Get Ground Truth Epsilon (Dynamics Params)
            # 从 env 的 buffer 中直接获取，确保与当前环境对齐
            # 假设 env.dynamics_params_buf 已经在 reset 时被填充
            if hasattr(env, 'dynamics_params_buf') and env.dynamics_params_buf is not None:
                epsilon = env.dynamics_params_buf.clone()
            else:
                # Fallback: 尝试从 extras 获取 (取决于 env 实现)
                # 注意：step() 之后 extras 才会刷新，第一帧可能为空，这里做个简单的容错
                epsilon = torch.zeros(env.num_envs, 4, device=device) 
            
            # Record
            z_history.append(mu_z.cpu())
            epsilon_history.append(epsilon.cpu())
            
            # 3. Step Environment
            # 在推理模式下，我们通常使用确定性策略 act_inference
            actions = policy.act_inference(obs)
            obs, _, _, _, infos = env.step(actions)
            
            # 如果 step 后 info 里有 dynamic_params，用它覆盖刚才的（更准确）
            if 'dynamic_params' in infos:
                epsilon_history[-1] = infos['dynamic_params'].cpu()

    # Stack into tensors
    # Shape: [Time, Batch, Dim]
    z_tensor = torch.stack(z_history, dim=0) 
    eps_tensor = torch.stack(epsilon_history, dim=0)
    
    return z_tensor, eps_tensor

def analyze_consistency(z_tensor):
    """
    验证目标 1: 单个 episode 内，z 是否处在一片小区域内。
    指标: 平均标准差 (Average Standard Deviation over Time)
    """
    # z_tensor: [Time, Batch, Z_Dim]
    
    # 计算每个环境在时间轴上的标准差
    # std_over_time: [Batch, Z_Dim]
    std_over_time = torch.std(z_tensor, dim=0)
    
    # 对所有维度和所有环境求平均
    avg_std = torch.mean(std_over_time).item()
    
    # 计算 latent 空间的平均尺度（用于对比）
    # 全局标准差
    global_std = torch.std(z_tensor.reshape(-1, z_tensor.shape[-1]), dim=0).mean().item()
    
    ratio = avg_std / (global_std + 1e-6)
    
    print(f"\n[Consistency Analysis]")
    print(f"  > Intra-episode Std (Stability): {avg_std:.4f} (Lower is better)")
    print(f"  > Global Std (Diversity):        {global_std:.4f}")
    print(f"  > Noise-to-Signal Ratio:         {ratio:.2%} (Should be low, e.g. < 10%)")
    
    return avg_std, ratio

def analyze_proximity_and_structure(z_tensor, eps_tensor):
    """
    验证目标 2: 相近的 epsilon 应该能得到相近的 z。
    指标: 距离相关系数 (Distance Correlation)
    """
    # 取时间平均，得到每个 episode 的代表性 z 和 epsilon
    # z_mean: [Batch, Z_Dim]
    z_mean = torch.mean(z_tensor, dim=0).numpy()
    eps_mean = torch.mean(eps_tensor, dim=0).numpy()
    
    # 1. 定量分析: 距离相关性
    # 计算 epsilon 的成对距离矩阵
    dist_eps = pdist(eps_mean, metric='euclidean')
    # 计算 z 的成对距离矩阵
    dist_z = pdist(z_mean, metric='euclidean')
    
    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(dist_eps, dist_z)
    
    print(f"\n[Proximity & Structure Analysis]")
    print(f"  > Distance Correlation (rho):    {correlation:.4f} (Closer to 1.0 is better)")
    print(f"    (rho > 0.5 implies strong structural alignment)")

    return z_mean, eps_mean, correlation

def visualize_latent_space(z_mean, eps_mean):
    """
    可视化: PCA 投影并按物理参数染色
    """
    batch_size, eps_dim = eps_mean.shape
    
    # PCA 降维到 2D
    reducer = PCA(n_components=2)
    z_2d = reducer.fit_transform(z_mean)
    
    # 假设 eps_mean 的前两维是 friction 和 mass (根据 g1_m_env.py 的实现)
    # dynamics_params_buf 顺序: [friction, mass, motor_f, motor_d]
    param_names = ['Friction', 'Mass', 'Motor Friction', 'Motor Damping']
    
    # 绘图
    fig, axes = plt.subplots(1, min(eps_dim, 4), figsize=(5 * min(eps_dim, 4), 5))
    if eps_dim == 1: axes = [axes]
    
    for i in range(min(eps_dim, 4)): # 最多画前4个参数
        ax = axes[i]
        param_values = eps_mean[:, i]
        
        # 散点图
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=param_values, cmap='viridis', alpha=0.7, s=20)
        ax.set_title(f"Colored by {param_names[i] if i < 4 else f'Param {i}'}")
        ax.set_xlabel("PC 1")
        if i == 0: ax.set_ylabel("PC 2")
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    plt.show()

def run_verification(env, policy, device='cuda'):
    """
    主入口函数
    """
    print("Starting Z-Space Verification...")
    
    # 1. Collect
    z_tensor, eps_tensor = collect_verification_data(env, policy, device=device)
    
    # 2. Consistency (Stability)
    analyze_consistency(z_tensor)
    
    # 3. Proximity (Structure)
    z_mean, eps_mean, corr = analyze_proximity_and_structure(z_tensor, eps_tensor)
    
    # 4. Visualization
    try:
        visualize_latent_space(z_mean, eps_mean)
        print("\nVisualization generated. Check plots.")
    except Exception as e:
        print(f"\nVisualization failed (maybe remote server?): {e}")

if __name__ == "__main__":
    # 这是一个示例用法，你需要将其集成到你的 play.py 中
    print("Please import `run_verification` in your play.py script and call it with your env and policy.")
    print("Example:")
    print("  from verify_z_structure import run_verification")
    print("  run_verification(env, agent.actor_critic)")