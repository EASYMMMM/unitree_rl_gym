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
                 device='cpu',
                 num_recon_observations = 29
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

    def init_storage(self, num_envs, num_steps, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageCVAE(num_envs, num_steps,
                                          actor_obs_shape, critic_obs_shape, action_shape,
                                          self.device, num_recon_observations = self.num_recon_observations)

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
        self.last_cvae_loss = None
        self.last_vt_recon_loss = None
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        ac = self.actor_critic
        std_param = ac.std  # [act_dim]

        for (obs_b, critic_obs_b, actions_b, target_v_b, adv_b, ret_b,
            old_logp_b, old_mu_b, old_sigma_b, hid_b, masks_b,
            next_obs_b, vt_tgt_b) in generator:

            # ========= (A) Update PPO  =========
            vt, z, mu_z, logstd_z = ac.encoder(obs_b)    # 建图，但随即对 PPO 阶段阻断
            vt = vt.detach()
            z  = z.detach()
            mu_z = mu_z.detach()

            mu = ac.policy_cvae(obs_b, vt, mu_z) # 只对 policy head 建图
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

            if has_recon or has_vt:
                vt2, z2, mu_z2, logstd_z2 = ac.encoder(obs_b)  # 新的图B
                losses = []
                obs_recon_loss = None
                vt_recon_loss = None

                if has_recon:
                    next_hat = ac.decoder(vt2, z2)
                    recon = F.mse_loss(next_hat, next_obs_b)
                    losses.append(self.recon_weight * recon)
                    obs_recon_loss = recon.item()
                if has_vt:
                    vt_loss = F.mse_loss(vt2, vt_tgt_b)
                    losses.append(self.vt_weight * vt_loss)
                    vt_recon_loss = vt_loss.item()

                # KL 始终可算（若你想只在 has_recon 时算，也可加条件）
                kl = 0.5 * torch.sum(torch.exp(2.0 * logstd_z2) + mu_z2.pow(2) - 1.0 - 2.0 * logstd_z2, dim=-1).mean()
                cvae_kl_loss = kl.item()
                losses.append(self.kl_weight * kl)

                cvae_loss = sum(losses)

                self.optim_cvae.zero_grad(set_to_none=True)
                cvae_loss.backward()
                nn.utils.clip_grad_norm_(itertools.chain(ac.encoder.parameters(), ac.decoder.parameters()),
                                        self.max_grad_norm)
                self.optim_cvae.step()
                self.last_cvae_loss = cvae_loss.item()
                # 速度重构损失优先记录vt_loss（如果有），否则记录recon（如果有）
                self.last_vt_recon_loss = vt_recon_loss
                self.last_obs_recon_loss = obs_recon_loss
                self.last_cvae_kl_loss = cvae_kl_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss


