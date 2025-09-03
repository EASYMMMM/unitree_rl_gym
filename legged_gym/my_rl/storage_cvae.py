# my_rl/storage_cvae.py
import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageCVAE:
    class Transition:
        def __init__(self):
            # 与原版一致
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            # NEW
            self.next_observations = None   # [N, obs_dim] (o_{t+1} 或 Δo)
            self.vt_target = None           # [N, vt_dim]  (可选)

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        T, N = num_transitions_per_env, num_envs
        obs_dim = obs_shape[0]
        act_dim = actions_shape[0]

        # Core（与原版一致的布局与 dtype）
        self.observations = torch.zeros(T, N, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(T, N, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(T, N, 1, device=self.device)
        self.actions = torch.zeros(T, N, *actions_shape, device=self.device)
        self.dones = torch.zeros(T, N, 1, device=self.device).byte()

        # PPO 需要的
        self.actions_log_prob = torch.zeros(T, N, 1, device=self.device)
        self.values = torch.zeros(T, N, 1, device=self.device)
        self.returns = torch.zeros(T, N, 1, device=self.device)
        self.advantages = torch.zeros(T, N, 1, device=self.device)
        self.mu = torch.zeros(T, N, *actions_shape, device=self.device)
        self.sigma = torch.zeros(T, N, *actions_shape, device=self.device)

        # NEW: CVAE 需要的
        self.next_observations = torch.zeros(T, N, obs_dim, device=self.device)  # o_{t+1} 或 Δo
        self.vt_targets = None  # 惰性创建（直到第一次 add_transitions 带来了 vt_target）

        self.num_transitions_per_env = T
        self.num_envs = N

        # RNN
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        # 与原版一致
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        # NEW: 额外存
        self.next_observations[self.step].copy_(transition.next_observations)
        if transition.vt_target is not None:
            if self.vt_targets is None:
                self.vt_targets = torch.zeros(self.num_transitions_per_env,
                                              self.num_envs,
                                              transition.vt_target.shape[-1],
                                              device=self.device,
                                              dtype=transition.vt_target.dtype)
            self.vt_targets[self.step].copy_(transition.vt_target)

        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    # 与原版完全一致（GAE & 归一化）
    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    # -------- 非 RNN mini-batch --------
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        assert batch_size % num_mini_batches == 0, \
            f"batch_size {batch_size} not divisible by num_mini_batches {num_mini_batches}"
        mini_batch_size = batch_size // num_mini_batches

        # ---- flatten 一次（减少重复视图 & 显存峰值）----
        observations = self.observations.flatten(0, 1)                        # [T*N, obs_dim]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)  # [T*N, cobs_dim]
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)                                   # [T*N, act_dim]
        values = self.values.flatten(0, 1)                                     # [T*N, 1]
        returns = self.returns.flatten(0, 1)                                   # [T*N, 1]
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)             # [T*N, 1]
        advantages = self.advantages.flatten(0, 1)                             # [T*N, 1]
        old_mu = self.mu.flatten(0, 1)                                         # [T*N, act_dim]
        old_sigma = self.sigma.flatten(0, 1)                                   # [T*N, act_dim]

        # NEW: next_obs / vt_targets 可能在 CPU，也可能在 GPU
        next_obs_flat = self.next_observations.flatten(0, 1)                   # [T*N, obs_dim]
        has_vt = self.vt_targets is not None
        vt_flat = (self.vt_targets.flatten(0, 1) if has_vt else None)          # [T*N, vt_dim] or None

        for _ in range(num_epochs):
            # 每个 epoch 重新洗牌（更稳定）
            # 如果 next_obs_flat 在 CPU，就用 CPU indices；否则用 GPU indices
            idx_device = next_obs_flat.device  # 让 indices 与被索引张量同设备
            indices = torch.randperm(batch_size, device=idx_device)

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mb = indices[start:end]

                # ---- 取出 batch；保证张量最终在 self.device（通常是 GPU）----
                # 观测/优势等主干张量通常在 GPU；如果 indices 在 CPU，就再转一次
                if observations.device != idx_device:
                    # 这种情况少见（通常 observations 和 idx_device 都是 cuda），留个保险
                    mb_gpu = mb.to(observations.device)
                else:
                    mb_gpu = mb

                obs_batch = observations[mb_gpu]
                critic_obs_batch = critic_observations[mb_gpu]
                actions_batch = actions[mb_gpu]
                target_values_batch = values[mb_gpu]
                returns_batch = returns[mb_gpu]
                old_actions_log_prob_batch = old_actions_log_prob[mb_gpu]
                advantages_batch = advantages[mb_gpu]
                old_mu_batch = old_mu[mb_gpu]
                old_sigma_batch = old_sigma[mb_gpu]

                # next_obs / vt：可能在 CPU，按需搬到 self.device
                next_obs_batch = next_obs_flat[mb]
                if next_obs_batch.device != self.device:
                    next_obs_batch = next_obs_batch.to(self.device, non_blocking=True)

                if has_vt:
                    vt_tgt_batch = vt_flat[mb]
                    if vt_tgt_batch.device != self.device:
                        vt_tgt_batch = vt_tgt_batch.to(self.device, non_blocking=True)
                else:
                    vt_tgt_batch = None

                yield (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
                    returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                    (None, None), None,
                    next_obs_batch, vt_tgt_batch)


    # -------- RNN mini-batch（与原版一致，末尾多两个项）--------
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        # NEW：把 next_obs / vt_targets 也做相同处理，保证与 obs 对齐
        padded_next_obs_trajectories, _ = split_and_pad_trajectories(self.next_observations, self.dones)
        has_vt = self.vt_targets is not None
        if has_vt:
            padded_vt_trajectories, _ = split_and_pad_trajectories(self.vt_targets, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                # NEW：对齐后的 next_obs / vt
                next_obs_batch = padded_next_obs_trajectories[:, first_traj:last_traj]
                vt_tgt_batch = (padded_vt_trajectories[:, first_traj:last_traj] if has_vt else None)

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # hidden states（与原版一致）
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [saved.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                               for saved in self.saved_hidden_states_a] if self.saved_hidden_states_a is not None else None
                hid_c_batch = [saved.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                               for saved in self.saved_hidden_states_c] if self.saved_hidden_states_c is not None else None
                hid_a_batch = hid_a_batch[0] if (hid_a_batch is not None and len(hid_a_batch) == 1) else hid_a_batch
                hid_c_batch = hid_c_batch[0] if (hid_c_batch is not None and len(hid_c_batch) == 1) else hid_c_batch

                yield (obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch,
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                       (hid_a_batch, hid_c_batch), masks_batch,
                       next_obs_batch, vt_tgt_batch)

                first_traj = last_traj
