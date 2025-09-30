from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.my_rl.actor_critic_cvae import ActorCriticCVAE

class G1_mRobot(LeggedRobot):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_stack_n = self.cfg.env.obs_stack_n
        self.priv_obs_stack_n = self.cfg.env.priv_obs_stack_n
        self._obs_stack_buf = None
        self._priv_stack_buf = None
        if self.headless == False:
            self._init_camera()

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
