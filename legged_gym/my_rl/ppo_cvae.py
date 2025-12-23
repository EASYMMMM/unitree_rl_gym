# my_rl/algorithms/ppo_cvae.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools

from .storage_cvae import RolloutStorageCVAE

class PPO_CVAE:
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 cvae_learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 recon_weight=1.0,
                 kl_weight=1e-3,
                 vt_weight=1.0,
                 z_smooth_weight=0.0,
                 geometry_weight=0.0, # [NEW] DPCVAE Isometric Loss Weight
                 device='cpu',
                 num_recon_observations=29
                 ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # Core
        self.actor_critic = actor_critic.to(device)
        self.transition = RolloutStorageCVAE.Transition()
        self.storage = None

        # optimizer
        rl_params = []
        rl_params += list(self.actor_critic.policy_cvae.parameters())  # 只更新策略头
        rl_params += list(self.actor_critic.critic.parameters())       # 价值网络
        rl_params += [self.actor_critic.std]                           # 动作噪声
        self.optim_rl = optim.Adam(rl_params, lr=learning_rate)

        cvae_params = itertools.chain(self.actor_critic.encoder.parameters(),
                                      self.actor_critic.decoder.parameters())
        self.optim_cvae = optim.Adam(cvae_params, lr=cvae_learning_rate)

        # PPO params
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # CVAE loss weights
        self.num_recon_observations = num_recon_observations
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.vt_weight = vt_weight
        self.geometry_weight = geometry_weight # [NEW]
        
        # Debug flag
        self._has_warned_missing_epsilon = False

    def init_storage(self, num_envs, num_steps, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageCVAE(num_envs, num_steps,
                                          actor_obs_shape, critic_obs_shape, action_shape,
                                          self.device, num_recon_observations=self.num_recon_observations)

    def test_mode(self):  self.actor_critic.test()
    def train_mode(self): self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, vt_target=None, dynamic_params=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # CVAE
        if next_obs is not None:
            self.transition.next_observations = next_obs
        if vt_target is not None:
            self.transition.vt_target = vt_target
        
        # [NEW] DPCVAE: Store dynamics params (epsilon)
        if dynamic_params is not None:
            self.transition.dynamic_params = dynamic_params

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # [NEW] Isometric Mapping Loss
    @staticmethod
    def geometry_preserving_loss(mu_z, epsilon):
        """
        让 Latent Space (mu_z) 的几何结构 模仿 Dynamics Params Space (epsilon) 的几何结构
        Loss = MSE( Normalize(Dist(mu_z)), Normalize(Dist(epsilon)) )
        """
        # 1. 计算 mu_z 的距离矩阵 (Batch, Batch)
        # p=2 (Euclidean Distance)
        dist_z = torch.cdist(mu_z, mu_z, p=2)
        
        # 2. 计算 epsilon 的距离矩阵 (Batch, Batch)
        # epsilon 已经是归一化过的，不需要梯度
        with torch.no_grad():
            dist_eps = torch.cdist(epsilon, epsilon, p=2)
            
        # 3. 归一化两个距离矩阵，使其具有可比性
        # 避免数值尺度影响 (例如 z 分布在 0-1，epsilon 分布在 0-1，但量级可能不同)
        # 加 1e-6 防止除零
        dist_z_norm = dist_z / (dist_z.mean() + 1e-6)
        dist_eps_norm = dist_eps / (dist_eps.mean() + 1e-6)
        
        # 4. 结构对齐损失 (Structural Alignment)
        loss = F.mse_loss(dist_z_norm, dist_eps_norm)
        return loss

    def update(self):
        mean_value_loss, mean_surrogate_loss = 0.0, 0.0
        self.last_cvae_loss = None
        self.last_vt_recon_loss = None
        self.last_obs_recon_loss = None
        self.last_cvae_kl_loss = None
        # [MODIFIED] 如果开启了几何权重，至少初始化为 0.0，确保 Tensorboard 能记录到
        self.last_geometry_loss = 0.0 if self.geometry_weight > 0 else None

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        ac = self.actor_critic
        std_param = ac.std  # [act_dim]

        for (obs_b, critic_obs_b, actions_b, target_v_b, adv_b, ret_b,
             old_logp_b, old_mu_b, old_sigma_b, hid_b, masks_b,
             next_obs_b, vt_tgt_b, epsilon_b) in generator: # [NEW] Unpack epsilon_b

            # ========= (A) Update PPO  =========
            vt, z, mu_z, logstd_z = ac.encoder(obs_b)    # 建图，但随即对 PPO 阶段阻断
            vt = vt.detach()
            z  = z.detach()
            mu_z = mu_z.detach()

            # [FIXED] PolicyHead 内部已处理 DPCVAE 的 obs 裁剪，所以这里直接传 obs_b 是安全的
            mu = ac.policy_cvae(obs_b, vt, mu_z)  # 只对 policy head 建图
            std = std_param.expand_as(mu)
            std = std.clamp_min(1e-3)
            dist = torch.distributions.Normal(mu, std)

            actions_log_prob_b = dist.log_prob(actions_b).sum(dim=-1, keepdim=True)
            entropy_b = dist.entropy().sum(dim=-1, keepdim=True)
            value_b = ac.critic(critic_obs_b)  # 只对 critic 建图

            # KL 自适应学习率
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(std / (old_sigma_b + 1e-8) + 1e-5) +
                        (old_sigma_b.pow(2) + (old_mu_b - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5,
                        dim=-1
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optim_rl.param_groups:
                        g['lr'] = self.learning_rate

            # PPO 损失
            ratio = torch.exp(actions_log_prob_b - old_logp_b)     # [B,1]
            surr1 = -adv_b * ratio
            surr2 = -adv_b * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surr1, surr2).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_v_b + (value_b - target_v_b).clamp(-self.clip_param, self.clip_param)
                v_loss = torch.max((value_b - ret_b).pow(2), (value_clipped - ret_b).pow(2)).mean()
            else:
                v_loss = (ret_b - value_b).pow(2).mean()

            rl_loss = surrogate_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_b.mean()

            # 反传 & step （只更新 policy_head + critic + log_std）
            self.optim_rl.zero_grad(set_to_none=True)
            rl_loss.backward()
            nn.utils.clip_grad_norm_(list(ac.policy_cvae.parameters()) +
                                     list(ac.critic.parameters()) + [ac.std],
                                     self.max_grad_norm)
            self.optim_rl.step()

            mean_value_loss     += v_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # ========= (B) Update CVAE =========
            has_recon = next_obs_b is not None
            has_vt    = vt_tgt_b is not None
            
            # [NEW] 是否计算几何损失: 需要有 epsilon 且权重 > 0
            has_geometry = (epsilon_b is not None) and (self.geometry_weight > 0)
            
            # [DEBUG] 如果开启了几何损失但没有收到数据，打印警告
            if self.geometry_weight > 0 and epsilon_b is None and not self._has_warned_missing_epsilon:
                print(f"[PPO_CVAE Warning] geometry_weight is {self.geometry_weight} but 'epsilon_b' (dynamic params) is None!")
                self._has_warned_missing_epsilon = True

            if has_recon or has_vt or has_geometry:
                # 重新计算当前 obs 的 latent (带梯度)
                vt2, z2, mu_z2, logstd_z2 = ac.encoder(obs_b)
                losses = []
                obs_recon_loss = None
                vt_recon_loss = None
                geometry_loss = None

                # 1. 重构损失
                if has_recon:
                    if getattr(ac, 'cvae_type', 'cvae') == 'dpcvae':
                        next_hat = ac.decoder(obs_b, z2)
                    else:
                        next_hat = ac.decoder(vt2, z2)
                        
                    recon = F.smooth_l1_loss(next_hat, next_obs_b, beta=0.05)
                    losses.append(self.recon_weight * recon)
                    obs_recon_loss = recon.item()
                
                # 2. 速度估计损失
                if has_vt:
                    vt_loss = F.mse_loss(vt2, vt_tgt_b)
                    losses.append(self.vt_weight * vt_loss)
                    vt_recon_loss = vt_loss.item()

                # 3. KL 散度
                kl = 0.5 * torch.sum(torch.exp(2.0 * logstd_z2) + mu_z2.pow(2) - 1.0 - 2.0 * logstd_z2, dim=-1).mean()
                cvae_kl_loss = kl.item()
                losses.append(self.kl_weight * kl)
                
                # 4. [NEW] Geometry Preserving Loss (Isometric Mapping)
                if has_geometry:
                    # [OPTIMIZATION] 随机采样以避免 O(N^2) 计算
                    batch_size = mu_z2.shape[0]
                    max_samples = 1024 # 限制采样数
                    
                    if batch_size > max_samples:
                        # 随机打乱并取前 max_samples 个
                        perm = torch.randperm(batch_size, device=self.device)[:max_samples]
                        sample_mu_z = mu_z2[perm]
                        sample_eps = epsilon_b[perm]
                    else:
                        sample_mu_z = mu_z2
                        sample_eps = epsilon_b
                        
                    geo_loss = self.geometry_preserving_loss(sample_mu_z, sample_eps)
                    losses.append(self.geometry_weight * geo_loss)
                    geometry_loss = geo_loss.item()

                # 总损失回传
                cvae_loss = sum(losses)

                self.optim_cvae.zero_grad(set_to_none=True)
                cvae_loss.backward()
                nn.utils.clip_grad_norm_(itertools.chain(ac.encoder.parameters(), ac.decoder.parameters()),
                                         self.max_grad_norm)
                self.optim_cvae.step()

                self.last_cvae_loss = cvae_loss.item()
                self.last_vt_recon_loss = vt_recon_loss
                self.last_obs_recon_loss = obs_recon_loss
                self.last_cvae_kl_loss = cvae_kl_loss
                
                # [MODIFIED] 更新几何损失记录 (如果本 batch 有值)
                if geometry_loss is not None:
                    self.last_geometry_loss = geometry_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss