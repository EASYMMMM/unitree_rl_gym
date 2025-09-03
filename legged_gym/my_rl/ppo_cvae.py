# my_rl/algorithms/ppo_cvae.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 recon_weight=1.0,
                 kl_weight=1e-3,
                 vt_weight=1.0,
                 device='cpu',
                 ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # Core
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCVAE.Transition()
        self.storage = None

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
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.vt_weight = vt_weight

    def init_storage(self, num_envs, num_steps, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageCVAE(num_envs, num_steps,
                                          actor_obs_shape, critic_obs_shape, action_shape,
                                          self.device)

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

    def process_env_step(self, rewards, dones, infos, next_obs=None, vt_target=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # NEW
        if next_obs is not None:
            self.transition.next_observations = next_obs
        if vt_target is not None:
            self.transition.vt_target = vt_target

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss, mean_surrogate_loss = 0.0, 0.0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (obs_b, critic_obs_b, actions_b, target_v_b, adv_b, ret_b,
            old_logp_b, old_mu_b, old_sigma_b, hid_b, masks_b,
            next_obs_b, vt_tgt_b) in generator:

            # ---------- 单次前向：Encoder + PolicyHead（不再调用 act() 与 cvae_aux_losses()）----------
            mu_b, std_b, vt_b, z_b, mu_z_b, logstd_z_b = self.actor_critic._actor_forward(obs_b)
            dist_b = torch.distributions.Normal(mu_b, std_b)

            # PPO 所需量：对“旧动作”求新分布下的 log_prob / 熵；Critic 前向不变
            actions_log_prob_b = dist_b.log_prob(actions_b).sum(-1, keepdim=True)
            entropy_b = dist_b.entropy().sum(dim=-1, keepdim=True)
            value_b = self.actor_critic.evaluate(
                critic_obs_b, masks=masks_b, hidden_states=(hid_b[1] if hid_b else None)
            )
 
            self.actor_critic.distribution = dist_b
            self.actor_critic._action_mean = mu_b
            self.actor_critic._action_std  = std_b
            self.actor_critic._entropy     = entropy_b

            # ---------- 自适应 KL 学习率（与原逻辑一致，用新 μ/σ 与 old μ/σ 计算 KL）----------
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(std_b / old_sigma_b + 1.e-5) +
                        (old_sigma_b.pow(2) + (old_mu_b - mu_b).pow(2)) / (2.0 * std_b.pow(2)) - 0.5,
                        dim=-1
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.learning_rate

            # ---------- PPO 主损失 ----------
            ratio = torch.exp(actions_log_prob_b - old_logp_b.squeeze(-1))
            surrogate = -adv_b.squeeze(-1) * ratio
            surrogate_clipped = -adv_b.squeeze(-1) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_v_b + (value_b - target_v_b).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_b - ret_b).pow(2)
                value_losses_clipped = (value_clipped - ret_b).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (ret_b - value_b).pow(2).mean()

            # ---------- CVAE 辅助损失（直接复用本次前向得到的 vt/z/μz/logσz）----------
            next_hat_b = self.actor_critic.decoder(vt_b, z_b)
            recon = F.mse_loss(next_hat_b, next_obs_b)
            kl_latent = 0.5 * torch.sum(
                torch.exp(2.0 * logstd_z_b) + mu_z_b.pow(2) - 1.0 - 2.0 * logstd_z_b, dim=-1
            ).mean()
            vt_loss = F.mse_loss(vt_b, vt_tgt_b)

            cvae_loss = self.recon_weight * recon + self.kl_weight * kl_latent + self.vt_weight * vt_loss

            # ---------- 总损失、反传、优化 ----------
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_b.mean() + cvae_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

