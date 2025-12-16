# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym import gymutil

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--record_video", "action": "store_true", "default": False, "help": "Record video of the environment. Overrides config file if provided."},
        
        # New Argument for Data Collection
        {"name": "--collect_data", "action": "store_true", "default": False, "help": "Collect data for offline RL (Stage 2)"},
        {"name": "--collect_max_steps", "type": int, "default": 10000, "help": "Max steps to collect if collect_data is True"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def play(args):
    # [Fix] 定义局部变量，不依赖全局变量
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FOLLOWED_CAMERA = False
    video_record = args.record_video # 使用小写变量名以区分

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # 这一步获取初始的 privileged_obs，用于提取 vt_target
    priv_obs = env.get_privileged_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # 视频录制相关
    VIDEO_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'video')
    os.makedirs(VIDEO_PATH, exist_ok=True)
    frames = []
    import imageio

    total_len = 10
    if video_record:
        total_len = 3
    
    # --- Data Collection Initialization ---
    COLLECT_DATA = args.collect_data
    
    # [Performance Fix] 如果开启大规模采集，强制关闭视频录制，防止 I/O 阻塞
    # 这里我们安全地修改局部变量 video_record
    if COLLECT_DATA and args.collect_max_steps > 2000 and video_record:
        print("[WARNING] Large scale data collection detected. Disabling video recording to improve performance.")
        video_record = False

    collected_data = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs_full': [],   # [NEW] 完整观测 (272 dim)
        'next_obs_recon': [],  # [NEW] 重构目标 (30 dim)
        'dones': [],
        'vt_targets': []
    }
    
    num_steps = 0
    max_steps = args.collect_max_steps if COLLECT_DATA else total_len * int(env.max_episode_length)
    
    print(f"Start simulation. Collect Data: {COLLECT_DATA}, Max Steps: {max_steps}")

    for i in range(max_steps):
        # 1. 保存当前观测 (Current Obs)
        current_obs = obs.clone()
        
        # 2. 获取 critic/privileged obs 用于提取 vt (与 runner_cvae.py 逻辑保持一致)
        # 如果 env 没有返回 priv_obs，则用 obs 代替
        current_critic_obs = priv_obs if priv_obs is not None else current_obs
        # 提取 vt_target (通常是 priv_obs 的前3维：线速度 target)
        vt_target = current_critic_obs[..., 0:3].clone()

        # 3. 策略推理
        actions = policy(obs.detach())
        
        # 4. 环境步进
        # 注意：接收 priv_obs (第二个返回值)，不再是用 _ 丢弃
        obs, priv_obs, rews, dones, infos = env.step(actions.detach())
        
        # 5. 数据采集逻辑
        if COLLECT_DATA:
            # A. 获取完整的下一帧 (用于 RL 计算)
            full_next_obs = obs.clone()

            # B. 获取重构目标 (用于 CVAE 监督)
            if 'obs_next_d' in infos:
                recon_next_obs = infos['obs_next_d'].clone()
            else:
                # 这种情况理论上不该发生
                recon_next_obs = torch.zeros(env.num_envs, 30, device=obs.device)

            # 将 Tensor 转到 CPU 并添加到列表 (避免显存爆炸)
            collected_data['obs'].append(current_obs.cpu())
            collected_data['actions'].append(actions.detach().cpu())
            collected_data['rewards'].append(rews.cpu().unsqueeze(-1)) # [N, 1]
            
            # [CHANGE] 分别存储
            collected_data['next_obs_full'].append(full_next_obs.cpu())
            collected_data['next_obs_recon'].append(recon_next_obs.cpu())
            
            collected_data['dones'].append(dones.cpu().unsqueeze(-1))  # [N, 1]
            collected_data['vt_targets'].append(vt_target.cpu())

            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Collected {num_steps} / {max_steps} steps...")

        # --- 以下是原有的可视化与日志逻辑 ---
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        
        # [Fix] 使用局部变量 video_record
        if video_record:
            if i % 2 == 0:
                filename = os.path.join(VIDEO_PATH, f"frame_{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                frames.append(filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        if FOLLOWED_CAMERA:
            env.set_follow_camera()

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    # --- 保存采集的数据 ---
    if COLLECT_DATA:
        print("Concatenating collected data...")
        # 拼接列表为 Tensor [Total_Steps * Num_Envs, Dim]
        dataset = {
            'observations': torch.cat(collected_data['obs'], dim=0),
            'actions': torch.cat(collected_data['actions'], dim=0),
            'rewards': torch.cat(collected_data['rewards'], dim=0),
            # [CHANGE] 保存两个 key
            'next_observations_full': torch.cat(collected_data['next_obs_full'], dim=0),
            'next_observations_recon': torch.cat(collected_data['next_obs_recon'], dim=0),
            'dones': torch.cat(collected_data['dones'], dim=0),
            'vt_targets': torch.cat(collected_data['vt_targets'], dim=0)
        }
        
        save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'data')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'offline_dataset_steps{max_steps}.pt')
        
        print(f"Saving dataset to {save_path} ...")
        torch.save(dataset, save_path)
        print(f"Dataset saved. Shape of observations: {dataset['observations'].shape}")

    # 推理结束后合成视频 [Fix] 使用 video_record
    if video_record and len(frames) > 0:
        video_file = os.path.join(VIDEO_PATH, 'play.mp4')
        imgs = [imageio.imread(f) for f in frames]
        imageio.mimsave(video_file, imgs, fps=25)
        print(f"[INFO] 视频已保存到: {video_file}")

if __name__ == '__main__':
    args = get_args()
    play(args)