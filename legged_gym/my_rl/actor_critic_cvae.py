import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic, get_activation


def mlp(sizes, activation="elu", out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(get_activation(activation))  # 直接append实例
        elif out_act is not None:
            layers.append(get_activation(out_act))     # 直接append实例
    return nn.Sequential(*layers)


# ----------------- CVAE Blocks -----------------
class Encoder(nn.Module):
    """ q(z_t | o_t) and v_hat_t = f(o_t) """
    def __init__(self, obs_dim, vt_dim, z_dim, hidden=(256, 256), activation="elu"):
        super().__init__()
        self.backbone = mlp((obs_dim,) + tuple(hidden), activation)
        last = hidden[-1] if hidden else obs_dim
        self.fc_vt = nn.Linear(last, vt_dim)
        self.fc_mu = nn.Linear(last, z_dim)
        self.fc_logstd = nn.Linear(last, z_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        vt = self.fc_vt(h)                                          # [B, vt_dim]
        mu_z = self.fc_mu(h)                                        # [B, z_dim]
        logstd_z = self.fc_logstd(h).clamp(-6.0, 2.0)               # 数值稳定
        std_z = torch.exp(logstd_z)
        # reparameterize
        z = mu_z + std_z * torch.randn_like(std_z)
        return vt, z, mu_z, logstd_z


class Decoder(nn.Module):
    """ p(o_{t+1} | z_t, v_hat_t) —— 先占位，下一步接重构损失 """
    def __init__(self, vt_dim, z_dim, obs_dim, hidden=(256, 256), activation="elu"):
        super().__init__()
        self.net = mlp((vt_dim + z_dim,) + tuple(hidden) + (obs_dim,), activation)

    def forward(self, vt, z):
        return self.net(torch.cat([vt, z], dim=-1))


class PolicyHead(nn.Module):
    """ π(a_t | o_t, v_hat_t, z_t) → μ """
    def __init__(self, obs_dim, vt_dim, z_dim, act_dim, hidden=(256, 256), activation="elu", num_single_obs=None):
        super().__init__()
        self.mu = mlp((obs_dim + vt_dim + z_dim,) + tuple(hidden) + (act_dim,), activation)
        self.num_single_obs = num_single_obs

    def forward(self, obs, vt, z):
        # [NEW] 如果传入的 obs 维度大于预期（全量历史），则在此处裁剪
        # 预期维度为 num_single_obs + 2 (相位)
        obs_in = obs
        if self.num_single_obs is not None:
             expected_dim = self.num_single_obs + 2
             if obs.shape[-1] > expected_dim:
                 # 取前 num_single_obs 维 (单帧) + 后 2 维 (相位)
                 obs_in = torch.cat([obs[:, :self.num_single_obs], obs[:, -2:]], dim=-1)

        x = torch.cat([obs_in, vt, z], dim=-1)
        return self.mu(x)


# 这是一个独立的 Q 函数网络：Q(s, a) -> value 
# 用于Uni-O4
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256, 256), activation="elu"):
        super().__init__()
        # 输入拼接: [obs, action]
        self.q = mlp((obs_dim + act_dim,) + tuple(hidden_dims) + (1,), activation)

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.q(x)


# ----------------- CVAE-ActorCritic -----------------
class ActorCriticCVAE(ActorCritic):
    """
    与 rsl_rl 基类接口对齐：
      - act(obs, masks=None, hidden_states=None) -> actions
           (并设置 .action_mean/.action_std/.entropy 供 PPO 使用)
      - get_actions_log_prob(actions) -> log_prob(actions | 当前分布)
      - evaluate(critic_obs, masks=None, hidden_states=None) -> values
      - act_inference(obs) -> deterministic action mean
    """
    is_recurrent = False

    def __init__(self,
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 # [NEW] 新增参数：单帧观测维度 (不含相位)
                 num_single_obs: int = None,
                 # 下面是 CVAE 的新增超参
                 vt_dim: int = 3,
                 z_dim: int = 16,
                 encoder_hidden_dims=(256, 256),
                 decoder_hidden_dims=(256, 256),
                 # 与父类保持一致的可配项
                 actor_hidden_dims=(256, 256, 256),
                 critic_hidden_dims=(256, 256, 256),
                 activation: str = "elu",
                 init_noise_std: float = 1.0,
                 num_recon_observations = 29,
                 **kwargs):
        # 先用父类构造（含 critic、可学习的 self.std 参数等）
        super().__init__(num_actor_obs=num_actor_obs,
                         num_critic_obs=num_critic_obs,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,  # 父类会建一个 actor，但我们不用它
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        # 保存形状/超参
        self.obs_dim = num_actor_obs
        self.act_dim = num_actions
        self.vt_dim = vt_dim
        self.z_dim = z_dim
        self.activation = activation

        # [NEW] 确定 PolicyHead 的输入维度
        # 如果传入了 num_single_obs，则策略输入 = 单帧 + 2(相位)
        if num_single_obs is not None:
            self.num_single_obs = num_single_obs
            self.policy_input_dim = num_single_obs + 2
        else:
            # 兼容旧逻辑：如果不传，默认使用全量观测
            self.num_single_obs = None
            self.policy_input_dim = num_actor_obs

        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))

        # CVAE-Actor 组件
        # Encoder 仍然接收 num_actor_obs (全量历史)
        self.encoder = Encoder(num_actor_obs, vt_dim, z_dim, encoder_hidden_dims, activation)
        self.decoder = Decoder(vt_dim, z_dim, num_recon_observations, decoder_hidden_dims, activation)  # 先挂上，下一步接损失
        
        # [MODIFIED] PolicyHead 使用 policy_input_dim, 并传入 num_single_obs 以便内部裁剪
        self.policy_cvae = PolicyHead(self.policy_input_dim, vt_dim, z_dim, num_actions, actor_hidden_dims, activation, num_single_obs=self.num_single_obs)

        # [NEW] 添加 Q-Critic 模块 (用于 Stage 2 离线 RL)
        # 关键修改：离线 Q 网络必须使用 Actor 观测 (num_actor_obs)，不能用特权观测 (num_critic_obs)
        # 因为在 CVAE 推演未来时，我们无法生成特权观测。
        self.q_critic = QNetwork(num_actor_obs, num_actions, critic_hidden_dims, activation)

        print('Encoder MLP:', {self.encoder})
        print('Decoder MLP:', {self.decoder})
        print(f'PolicyHead MLP (Input={self.policy_input_dim}):', {self.policy_cvae})
        print('Q-Critic MLP:', {self.q_critic})

        # 移除手动设置 self.actor = None，避免 JIT 混淆
        # self.actor 将保留为父类创建的默认 nn.Sequential (未使用)
        # 这不会影响 policy_cvae 的逻辑

        # 轻微正交初始化（与 rsl_rl 风格一致）
        # 这会自动初始化包括 q_critic 在内的所有子模块
        self.apply(self._init_weights_orthogonal)

    @staticmethod
    def _init_weights_orthogonal(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # --------- 内部：用 CVAE 生成策略分布参数 ---------
    def _actor_forward(self, observations):
        """
        返回：
          mu: [B, act_dim]
          std: [B, act_dim]（由父类的 self.std 展开）
          vt, z, mu_z, logstd_z: 便于后续扩展（当前 PPO 未用）
        """
        vt, z, mu_z, logstd_z = self.encoder(observations)
        
        # [MODIFIED] 直接传入 observations，由 PolicyHead 内部检测并裁剪
        mu = self.policy_cvae(observations, vt, mu_z)  # 动作均值
        std = self.std.expand_as(mu)                # 父类已注册为可学习噪声 std（形状 [act_dim]）
        return mu, std, vt, z, mu_z, logstd_z

    # --------- 与父类一致的公开接口 ---------
    @torch.no_grad()
    def act_inference(self, observations):
        """确定性动作（部署/评估）"""
        mu, _, _, _, _, _ = self._actor_forward(observations)
        return mu

    def act(self, observations, masks=None, hidden_states=None):
        """
        采样动作：PPO 在 rollout/更新时都会调用本函数，并期望：
          - 返回 actions
          - 同时在对象上缓存：self.action_mean / self.action_std / self.entropy
        """
        mu, std, _, _, _, _ = self._actor_forward(observations)
        std = std.clamp_min(1e-3)
        dist = Normal(mu, std)
        actions = dist.sample()
        # 缓存分布属性（PPO.update 会直接读）
        self._action_mean = mu
        self._action_std = std
        self._entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # 若父类有 self.distribution，可同步一下（可选）
        self.distribution = dist
        return actions

    def get_actions_log_prob(self, actions):
        """
        用“最近一次 act(observations)”缓存下来的分布计算 log_prob(actions)。
        PPO.update 里正是这么用的。
        """
        # 若希望更稳健，可在这里 assert 已经调用过 act() 并缓存了 distribution
        dist = getattr(self, "distribution", None)
        if dist is None:
            # 兜底：用缓存的均值与方差重建分布
            dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)
    
    @property
    def action_mean(self):
        return self._action_mean
    
    @property
    def action_std(self):
        return self._action_std

    @property
    def entropy(self):
        return self._entropy

    # 移除 property actor(self)，因为这会混淆 JIT 且并未使用。
    # 外部调用应使用 act_inference 或 policy_cvae。

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        仅做 critic 的评估（不更新），与 rsl_rl 的约定保持一致。
        这里直接复用父类里构造好的 self.critic。
        """
        return self.critic(critic_observations)

    def cvae_aux_losses(self,
                        observations: torch.Tensor,
                        next_observations: torch.Tensor = None,
                        vt_target: torch.Tensor = None,
                        recon_weight: float = 1.0,
                        kl_weight: float = 1e-3,
                        vt_weight: float = 1.0):
        """
        纯前向：给定 o_t（以及可选 o_{t+1}、v_t target），返回 recon/kl/vt 及合成的 cvae_loss。
        不做任何优化步骤；在 PPO.update 里与主损失相加即可。
        """
        # 复用 encoder 产生 (vt, z, mu_z, logstd_z)
        vt, z, mu_z, logstd_z = self.encoder(observations)

        # 重构损失（若提供 next_observations）
        if next_observations is not None:
            next_hat = self.decoder(vt, z)
            recon = F.mse_loss(next_hat, next_observations)
        else:
            recon = torch.tensor(0.0, device=observations.device)

        # KL(q||N(0,I))
        kl = 0.5 * torch.sum(torch.exp(2.0*logstd_z) + mu_z**2 - 1.0 - 2.0*logstd_z, dim=-1).mean()

        # 速度监督（可选）
        vt_loss = torch.tensor(0.0, device=observations.device)
        if vt_target is not None:
            vt_loss = F.mse_loss(vt, vt_target)

        cvae_loss = recon_weight * recon + kl_weight * kl + vt_weight * vt_loss
        return {
            "recon_loss": recon,
            "kl_loss": kl,
            "vt_loss": vt_loss,
            "cvae_loss": cvae_loss,
        }