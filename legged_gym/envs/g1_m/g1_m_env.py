from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
from legged_gym.my_rl.actor_critic_cvae import ActorCriticCVAE

class G1_mRobot(LeggedRobot):
    def __init__(self, *args,  **kwargs):
        # [FIX] 将变量初始化提到 super().__init__ 之前
        # 避免 super().__init__ -> _init_buffers 计算出的值被后续的 None 覆盖
        self.payloads = None 
        self.motor_strengths = None
        self.dynamics_params_buf = None 

        super().__init__(*args, **kwargs)
        
        self.obs_stack_n = self.cfg.env.obs_stack_n
        self.priv_obs_stack_n = self.cfg.env.priv_obs_stack_n
        self._obs_stack_buf = None
        self._priv_stack_buf = None
        
        # [REMOVED] 下面这几行被移到了 super 之前，防止覆盖
        # self.payloads = None 
        # self.motor_strengths = None
        # self.dynamics_params_buf = None 
        
        if self.headless == False:
            self._init_camera()

    # ----------------------------------------------------------------------
    # [NEW] 核心修改 1: 重写以捕获并存储 额外负载(Mass)
    # ----------------------------------------------------------------------
    def _process_rigid_body_props(self, props, env_id):
        # 1. 初始化存储容器 (仅在第0个环境时执行一次)
        if env_id == 0:
            self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            
        # 2. 应用随机化
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            props[0].mass += added_mass
            
            # 3. 记录真值
            self.payloads[env_id] = added_mass
            
        return props

    # ----------------------------------------------------------------------
    # [NEW] 核心修改 2: 支持 电机参数(Dof Props) 修改与存储
    # ----------------------------------------------------------------------
    def _process_dof_props(self, props, env_id):
        # 1. 务必先调用父类方法，处理 soft limits 等关键逻辑
        props = super()._process_dof_props(props, env_id)
        
        # 2. 初始化存储容器 [num_envs, 2] -> (friction, damping)
        if env_id == 0:
            self.motor_strengths = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        # 3. 应用电机参数随机化
        if self.cfg.domain_rand.randomize_motor_props:
            f_rng = self.cfg.domain_rand.motor_friction_range
            d_rng = self.cfg.domain_rand.motor_damping_range
            
            # 采样随机值
            rand_f = np.random.uniform(f_rng[0], f_rng[1])
            rand_d = np.random.uniform(d_rng[0], d_rng[1])
            
            # 应用到所有关节
            for i in range(len(props)):
                props["friction"][i] = rand_f
                props["damping"][i] = rand_d
                
            # 记录真值
            self.motor_strengths[env_id, 0] = rand_f
            self.motor_strengths[env_id, 1] = rand_d
        
        return props

    # ----------------------------------------------------------------------
    # [NEW] 核心修改 3: 缓存机制 - 只计算一次并缓存
    # ----------------------------------------------------------------------
    def _update_dynamics_params_buf(self):
        """在初始化结束后调用一次，计算并缓存归一化的 mu"""
        # 1. 摩擦力 (Ground Friction)
        # 修复 Device 问题：确保在 GPU 上
        if hasattr(self, 'friction_coeffs'):
            friction = self.friction_coeffs.view(-1).to(self.device)
            f_min, f_max = self.cfg.domain_rand.friction_range
            norm_friction = (friction - f_min) / (f_max - f_min + 1e-6)
        else:
            norm_friction = torch.zeros(self.num_envs, device=self.device)

        # 2. 负载质量 (Payload Mass)
        if self.payloads is not None:
            mass = self.payloads.view(-1).to(self.device)
            m_min, m_max = self.cfg.domain_rand.added_mass_range
            norm_mass = (mass - m_min) / (m_max - m_min + 1e-6)
        else:
            norm_mass = torch.zeros(self.num_envs, device=self.device)

        # 3. 电机参数 (Motor Props)
        if self.motor_strengths is not None and self.cfg.domain_rand.randomize_motor_props:
            mf_min, mf_max = self.cfg.domain_rand.motor_friction_range
            md_min, md_max = self.cfg.domain_rand.motor_damping_range
            
            norm_mf = (self.motor_strengths[:, 0].view(-1).to(self.device) - mf_min) / (mf_max - mf_min + 1e-6)
            norm_md = (self.motor_strengths[:, 1].view(-1).to(self.device) - md_min) / (md_max - md_min + 1e-6)
        else:
            norm_mf = torch.zeros(self.num_envs, device=self.device)
            norm_md = torch.zeros(self.num_envs, device=self.device)

        # 拼接: [friction, mass, motor_f, motor_d] -> (num_envs, 4)
        self.dynamics_params_buf = torch.stack([norm_friction, norm_mass, norm_mf, norm_md], dim=-1).clamp(0.0, 1.0)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
    
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        # [NEW] 初始化完成后，立刻计算并缓存 dynamics params
        self._update_dynamics_params_buf()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        if self.headless == False:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return
    
    def set_follow_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos

    def _post_physics_step_callback(self):
        self.update_feet_state()
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.compute_recon_obs()
        return super()._post_physics_step_callback()
    
    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        cur_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             self.commands[:, :3] * self.commands_scale,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.actions,
                             ), dim=-1)
        
        if self._obs_stack_buf is None:
            self._obs_stack_buf = torch.zeros(self.obs_stack_n, self.num_envs, cur_obs.shape[-1], device=cur_obs.device)

        self._obs_stack_buf = torch.roll(self._obs_stack_buf, shifts=1, dims=0)
        self._obs_stack_buf[0] = cur_obs

        stacked_obs = self._obs_stack_buf.permute(1, 0, 2).reshape(self.num_envs, -1)
        self.obs_buf = torch.cat((stacked_obs, sin_phase, cos_phase), dim=-1)
       
        cur_priv = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                        self.base_ang_vel  * self.obs_scales.ang_vel,
                        self.projected_gravity,
                        self.commands[:, :3] * self.commands_scale,
                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                        self.dof_vel * self.obs_scales.dof_vel,
                        self.actions,
                        sin_phase,
                        cos_phase
                        ),dim=-1)
        if not hasattr(self, '_priv_stack_buf') or self._priv_stack_buf is None:
            self._priv_stack_buf = torch.zeros(self.priv_obs_stack_n, self.num_envs, cur_priv.shape[-1], device=cur_priv.device)
        self._priv_stack_buf = torch.roll(self._priv_stack_buf, shifts=1, dims=0)
        self._priv_stack_buf[0] = cur_priv
        stacked_priv = self._priv_stack_buf.permute(1, 0, 2).reshape(self.num_envs, -1)
        self.privileged_obs_buf = stacked_priv
       
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # ----------------------------------------------------------------------
        # [NEW] 核心修改 4: 将归一化的动力学参数传出
        # 优化: 直接使用缓存，零计算开销
        # ----------------------------------------------------------------------
        self.extras['dynamic_params'] = self.dynamics_params_buf

    def compute_recon_obs(self):
        recon_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             ), dim=-1) # 30
        self.extras['obs_next_d'] = recon_obs

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._obs_stack_buf is not None:
            cur_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.commands[:, :3] * self.commands_scale,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions), dim=-1)   
            self._obs_stack_buf[:, env_ids, :] = 0.0
            self._obs_stack_buf[0][env_ids] = cur_obs[env_ids]
        if hasattr(self, '_priv_stack_buf') and self._priv_stack_buf is not None:
            self._priv_stack_buf[:, env_ids, :] = 0.0

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        return 1.0
    
    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)