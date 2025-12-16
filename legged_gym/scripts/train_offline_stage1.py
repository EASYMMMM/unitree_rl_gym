# ==============================================================================
# 1. 关键修复：必须最先导入 isaacgym，且在 torch 之前
# ==============================================================================
import isaacgym 
import torch
# ==============================================================================

import os
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import wandb 

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
# 确保导入路径正确
from legged_gym.my_rl.actor_critic_cvae import ActorCriticCVAE

# ---------------------------------------------------------
# 1. 自定义数据集类
# ---------------------------------------------------------
class OfflineDataset(Dataset):
    def __init__(self, data_path, device='cpu'):
        print(f"Loading offline data from: {data_path}")
        data = torch.load(data_path, map_location=device)
        
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.vt_targets = data['vt_targets']
        
        # [CHANGE] 加载两份 Next Obs
        self.next_observations_full = data['next_observations_full']   # 用于 RL (272维)
        self.next_observations_recon = data['next_observations_recon'] # 用于 CVAE (30维)
        
        self.critic_observations = self.observations 

        self.length = self.observations.shape[0]
        print(f"Data loaded. Total transitions: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.critic_observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.vt_targets[idx],
            self.next_observations_full[idx],  # [NEW]
            self.next_observations_recon[idx]  # [NEW]
        )

# ---------------------------------------------------------
# 2. Expectile Loss
# ---------------------------------------------------------
def expectile_loss(diff, tau=0.7):
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()

# ---------------------------------------------------------
# 3. Stage 1 训练主逻辑
# ---------------------------------------------------------
def train_offline_stage1(args):
    # ================= WandB Init =================
    run_name = f"Stage1_Offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="unitree_rl_offline",
        name=run_name,
        config=vars(args),
        mode="online"
    )

    # ================= 配置与初始化 =================
    # 1. 准备环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    device = args.rl_device
    batch_size = 256
    max_epochs = 100 
    save_interval = 10
    
    # CVAE 超参数
    recon_weight = 1.0
    kl_weight = 1e-4 
    vt_weight = 1.0
    
    # Critic 超参数 (IQL)
    gamma = 0.99
    tau = 0.7 
    q_lr = 3e-4
    cvae_lr = 1e-4
    
    # 路径设置
    experiment_name = train_cfg.runner.experiment_name
    data_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name, 'exported', 'data')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No .pt dataset found in {data_dir}")
    dataset_file = sorted(files)[-1]
    data_path = os.path.join(data_dir, dataset_file)
    
    # ================= 数据加载 =================
    dataset = OfflineDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ================= 模型维度获取 =================
    num_obs = env_cfg.env.num_observations
    num_privileged_obs = env_cfg.env.num_privileged_obs
    num_actions = env_cfg.env.num_actions
    
    if num_privileged_obs is None:
        num_privileged_obs = num_obs

    print(f"Model Dims: Obs={num_obs}, PrivObs={num_privileged_obs}, Actions={num_actions}")

    # ================= 模型构建 =================
    print("Building ActorCriticCVAE model...")
    policy_cfg = train_cfg.policy
    
    if not isinstance(policy_cfg, dict):
        policy_cfg_dict = vars(policy_cfg)
    else:
        policy_cfg_dict = policy_cfg

    # [Fix] 使用 recon 数据集来自动检测重构维度
    sample_next_obs_recon = dataset.next_observations_recon[0]
    real_recon_dim = sample_next_obs_recon.shape[0]
    print(f"Auto-detected reconstruction dim from dataset: {real_recon_dim}")

    actor_critic = ActorCriticCVAE(
        num_actor_obs=num_obs,
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        num_recon_observations=real_recon_dim, # 传入正确的 30 维
        **policy_cfg_dict
    ).to(device)
    
    # 加载预训练权重 (Source Policy)
    if args.checkpoint != -1:
        load_run = args.load_run if args.load_run else train_cfg.runner.run_name
        model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name, load_run, f"model_{args.checkpoint}.pt")
        print(f"Loading checkpoint from: {model_path}")
        
        state_dict = torch.load(model_path, map_location=device)['model_state_dict']
        
        model_dict = actor_critic.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        if len(pretrained_dict) < len(state_dict):
            print(f"[Warning] Some keys were ignored (e.g. missing q_critic). Loaded {len(pretrained_dict)}/{len(model_dict)} keys.")
        
        model_dict.update(pretrained_dict)
        actor_critic.load_state_dict(model_dict)
    else:
        print("[Info] No checkpoint loaded. Initializing from scratch (Not recommended for Stage 1).")

    # ================= Target Networks =================
    if not hasattr(actor_critic, 'q_critic'):
        raise AttributeError("ActorCriticCVAE instance has no attribute 'q_critic'. Please update your actor_critic_cvae.py.")

    target_q_critic = copy.deepcopy(actor_critic.q_critic)
    target_q_critic.to(device)
    target_q_critic.eval()

    # ================= 优化器 =================
    cvae_params = list(actor_critic.encoder.parameters()) + list(actor_critic.decoder.parameters())
    optimizer_cvae = optim.Adam(cvae_params, lr=cvae_lr)
    optimizer_q = optim.Adam(actor_critic.q_critic.parameters(), lr=q_lr)
    
    # ================= 训练循环 =================
    print(f"Starting Stage 1 Offline Training for {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        epoch_cvae_loss = 0.0
        epoch_q_loss = 0.0
        
        # [CHANGE] Unpack 8 个返回值
        for batch_idx, (obs, c_obs, action, reward, done, vt_target, next_obs_full, next_obs_recon) in enumerate(dataloader):
            
            # ---------------- Part A: CVAE Fine-tuning ----------------
            vt, z, mu_z, logstd_z = actor_critic.encoder(obs)
            next_obs_hat = actor_critic.decoder(vt, z)
            
            # [FIX] 计算重构损失时，使用 next_obs_recon (30维)
            # 现在 Decoder 输出 (30维) 和 Target (30维) 完美对齐
            recon_loss = F.mse_loss(next_obs_hat, next_obs_recon)
            vt_loss = F.mse_loss(vt, vt_target)
            kl_loss = 0.5 * torch.sum(torch.exp(2.0 * logstd_z) + mu_z.pow(2) - 1.0 - 2.0 * logstd_z, dim=-1).mean()
            
            total_cvae_loss = (recon_weight * recon_loss) + (vt_weight * vt_loss) + (kl_weight * kl_loss)
            
            optimizer_cvae.zero_grad()
            total_cvae_loss.backward()
            optimizer_cvae.step()
            
            epoch_cvae_loss += total_cvae_loss.item()
            
            # ---------------- Part B: Q-Critic Training (IQL) ----------------
            with torch.no_grad():
                # [FIX] 计算 Target Q 时，使用 next_obs_full (272维)
                # 因为 Actor 和 Critic 需要完整的状态信息
                next_action_mu = actor_critic.act_inference(next_obs_full)
                target_q_next = target_q_critic(next_obs_full, next_action_mu)
                target_q = reward + (1.0 - done.float()) * gamma * target_q_next
            
            current_q = actor_critic.q_critic(c_obs, action)
            
            diff = target_q - current_q
            q_loss = expectile_loss(diff, tau=tau)
            
            optimizer_q.zero_grad()
            q_loss.backward()
            optimizer_q.step()
            
            epoch_q_loss += q_loss.item()
            
            # 软更新 Target Q
            with torch.no_grad():
                alpha = 0.005
                for param, target_param in zip(actor_critic.q_critic.parameters(), target_q_critic.parameters()):
                    target_param.data.mul_(1 - alpha)
                    torch.add(target_param.data, param.data, alpha=alpha, out=target_param.data)

        avg_cvae_loss = epoch_cvae_loss / len(dataloader)
        avg_q_loss = epoch_q_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{max_epochs} | CVAE: {avg_cvae_loss:.4f} | Q-Critic: {avg_q_loss:.4f}")
        
        wandb.log({
            "Stage1/CVAE_Loss": avg_cvae_loss,
            "Stage1/Q_Critic_Loss": avg_q_loss,
            "Stage1/Epoch": epoch + 1
        })
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name, 'exported', f"stage1_model_{epoch+1}.pt")
            torch.save(actor_critic.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    wandb.finish()
    print("Stage 1 Training Complete.")

if __name__ == '__main__':
    args = get_args()
    train_offline_stage1(args)