from rsl_rl.runners import OnPolicyRunner
import os
import copy
from ..utils.wandb_writer import WandbSummaryWriter
from datetime import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from .ppo_cvae import PPO_CVAE
from .actor_critic_cvae import ActorCriticCVAE
import torch

import time
import os
from collections import deque
import statistics

class OnPolicyRunner_WB(OnPolicyRunner):
    """
    Wrapper of OnPolicyRunner that mirrors all TensorBoard logs to Weights & Biases via WandbSummaryWriter.
    只在 rank0 写 wandb（WandbSummaryWriter 内部已处理）。
    """

    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 use_wandb=True,
                 wandb_project="unitree_rl",
                 wandb_run_name=None,
                 wandb_mode="online",      # "online" | "offline" | "disabled"
                 wandb_entity=None,
                 wandb_extra_config: dict = None):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs

        # 通过 eval() 创建策略与算法；确保上面 import 了 ActorCriticCVAE / PPO_CVAE
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic / ActorCriticCVAE
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO / PPO_CVAE
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model（PPO / PPO_CVAE 会各自创建 storage）
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        self._wb_enabled   = use_wandb
        self._wb_project   = wandb_project
        self._wb_run_name  = train_cfg["runner"]["run_name"] + '-'+ train_cfg["runner"]["experiment_name"]
        self._wb_mode      = wandb_mode
        self._wb_entity    = wandb_entity
        self._wb_cfg_extra = copy.deepcopy(wandb_extra_config) if wandb_extra_config else {}

        # 写入到 wandb.config 的信息
        self._wb_base_config = {
            "runner": {k: v for k, v in self.cfg.items() if isinstance(v, (int, float, str, bool))},
            "algorithm": {k: v for k, v in self.alg_cfg.items() if isinstance(v, (int, float, str, bool))},
            "policy": {k: v for k, v in self.policy_cfg.items() if isinstance(v, (int, float, str, bool))},
            "env": {
                "num_envs": getattr(self.env, "num_envs", None),
                "num_obs": getattr(self.env, "num_obs", None),
                "num_privileged_obs": getattr(self.env, "num_privileged_obs", None),
                "num_actions": getattr(self.env, "num_actions", None),
                "max_episode_length": getattr(self.env, "max_episode_length", None),
            },
        }
        self._wb_base_config.update(self._wb_cfg_extra)

    def _ensure_wandb_writer(self):
        if self.log_dir is None or self.writer is not None:
            return
        time_stamp = datetime.now().strftime("%d-%H-%M-%S")
        base_run_name = self._wb_run_name
        composed_run_name = f"{base_run_name}-{time_stamp}"
        self.writer = WandbSummaryWriter(
            log_dir=self.log_dir,
            use_wandb=self._wb_enabled,
            project=self._wb_project,
            run_name=composed_run_name,
            config=self._wb_base_config,
            mode=self._wb_mode,
            entity=self._wb_entity,
            flush_secs=10,
        )

    # -------- 覆盖 learn：与原版一致，只在 rollout 阶段多传 next_obs / vt_target --------
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._ensure_wandb_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 初始观测
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # train 模式

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)


        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs_next, privileged_obs_next, rewards, dones, infos = self.env.step(actions)
                    critic_obs_next = privileged_obs_next if privileged_obs_next is not None else obs_next
                    # TODO: vt or vt+1
                    # vt_target = privileged_obs_next[..., 0:3]  # vt+1
                    vt_target = critic_obs[..., 0:3]     # vt
                    # 传入 PPO / PPO_CVAE：把 next_obs 与 vt_target 一并记录
                    obs_next_d = obs_next.to(self.device)
                    critic_obs_next_d = critic_obs_next.to(self.device)
                    try:
                        # PPO_CVAE  
                        self.alg.process_env_step(
                            rewards.to(self.device),
                            dones.to(self.device),
                            infos,
                            next_obs=infos['obs_next_d'],
                            vt_target=(vt_target.to(self.device) if vt_target is not None else None),
                        )
                    except KeyError:
                        # PPO
                        self.alg.process_env_step(
                            rewards.to(self.device),
                            dones.to(self.device),
                            infos,
                        )
                   
                    obs, critic_obs = obs_next_d, critic_obs_next_d

                    # log
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards.to(self.device)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.numel() > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # GAE/Returns
                start = stop
                self.alg.compute_returns(critic_obs)

            # -------- 学习一步（PPO 或 PPO_CVAE）--------
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # log
            if self.log_dir is not None:
                # 兼容CVAE PPO，记录cvae_loss和vt_recon_loss
                cvae_loss = getattr(self.alg, "last_cvae_loss", None)
                vt_recon_loss = getattr(self.alg, "last_vt_recon_loss", None)
                obs_recon_loss = getattr(self.alg, "last_obs_recon_loss", None)
                cvae_kl_loss = getattr(self.alg, "last_kl_loss", None)
                if cvae_loss is not None:
                    self.writer.add_scalar('Loss/cvae_loss', cvae_loss, it)
                if vt_recon_loss is not None:
                    self.writer.add_scalar('Loss/vt_recon_loss', vt_recon_loss, it)
                if obs_recon_loss is not None:
                    self.writer.add_scalar('Loss/obs_recon_loss', obs_recon_loss, it)
                if cvae_kl_loss is not None:
                    self.writer.add_scalar('Loss/cvae_kl_loss', cvae_kl_loss, it)
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    # 兼容原版AC和CVAE算法的保存
    def save(self, path, infos=None):
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        # 原版PPO有optimizer，CVAE有optim_rl/optim_cvae
        if hasattr(self.alg, 'optimizer'):
            save_dict['optimizer_state_dict'] = self.alg.optimizer.state_dict()
        else:
            if hasattr(self.alg, 'optim_rl'):
                save_dict['optim_rl_state_dict'] = self.alg.optim_rl.state_dict()
            if hasattr(self.alg, 'optim_cvae'):
                save_dict['optim_cvae_state_dict'] = self.alg.optim_cvae.state_dict()
        torch.save(save_dict, path)

    # 兼容原版AC和CVAE算法的加载
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # [Fix] 添加 strict=False
        # 这样即使 checkpoint 里缺少 q_critic 的权重（或者多了其他权重），也不会报错。
        # 对于 play.py，q_critic 会保持随机初始化，但这不影响数据采集，因为我们只用 Actor。
        self.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 原版PPO有optimizer，CVAE有optim_rl/optim_cvae
        if hasattr(self.alg, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if hasattr(self.alg, 'optim_rl') and 'optim_rl_state_dict' in checkpoint:
                self.alg.optim_rl.load_state_dict(checkpoint['optim_rl_state_dict'])
            if hasattr(self.alg, 'optim_cvae') and 'optim_cvae_state_dict' in checkpoint:
                self.alg.optim_cvae.load_state_dict(checkpoint['optim_cvae_state_dict'])
        
        if 'iter' in checkpoint:
            self.current_learning_iteration = checkpoint['iter']
        
        # 如果是部分加载（比如 strict=False），这里打印个提示会更好，不过不是必须的
        print(f"Loaded model from {path} (strict=False)")
        
        return checkpoint.get('infos', None)