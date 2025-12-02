import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from typing import Dict
import random
import math
# env related
from envs.base_task import BaseTask

# utils
from utils.terrain import Terrain
from utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift, CubicSpline
from utils.helpers import class_to_dict
import torchvision
import cv2

# config
from configs import LeggedRobotCfg
from global_config import ROOT_DIR
from utils.utils import random_quat

class Tron1Robot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
    
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_cost_function()
        self.init_done = True
        self.global_counter = 0

        # self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.post_physics_step()

    def compute_dof_vel(self):
        diff = (
            torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi)
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt

        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        self.last_dof_pos[:] = self.dof_pos[:]

    #------------ enviorment core ----------------
    def _init_buffers(self):
  
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.foot_velocities = self.rigid_body_states[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_states[:, self.feet_indices,
                              0:3]
        self.foot_heights = torch.zeros_like(self.foot_positions[:, :, 2])
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 2, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        str_rng = self.cfg.domain_rand.motor_strength_range
        kp_str_rng = self.cfg.domain_rand.kp_range
        kd_str_rng = self.cfg.domain_rand.kd_range
        deadband_rng = self.cfg.domain_rand.deadband_range

        #修改：对同一个环境的所有关节取相同的值
        rand_strength = torch.rand(self.num_envs, 1, device=self.device)
        self.motor_strength = (str_rng[1] - str_rng[0]) * rand_strength + str_rng[0]
        self.motor_strength = self.motor_strength.expand(-1, self.num_actions)
        # 每个环境生成一组 kp/kd/motor_strength，然后复制到每个 action 上
        rand_kp = torch.rand(self.num_envs, 1, device=self.device)
        self.kp_factor = (kp_str_rng[1] - kp_str_rng[0]) * rand_kp + kp_str_rng[0]
        self.kp_factor = self.kp_factor.expand(-1, self.num_actions)  # shape: [num_envs, num_actions]
        rand_kd = torch.rand(self.num_envs, 1, device=self.device)
        self.kd_factor = (kd_str_rng[1] - kd_str_rng[0]) * rand_kd + kd_str_rng[0]
        self.kd_factor = self.kd_factor.expand(-1, self.num_actions)  # shape: [num_envs, num_actions]

        rand_deadband = torch.rand(self.num_envs, 1, device=self.device)
        self.deadband_factor = (deadband_rng[1] - deadband_rng[0]) * rand_deadband + deadband_rng[0]
        self.deadband_factor = self.deadband_factor.expand(-1, self.num_actions)

        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.env.history_encoding:
             self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)#vx,w,height
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.last_contacts_nofly = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            dtype=torch.bool,
            device=self.device
        )
        self.stable_contact_counter = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device
        )
      
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            print(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[:,i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
    
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
            
        self.lag_buffer = torch.zeros(self.num_envs,self.cfg.domain_rand.lag_timesteps,self.num_actions,device=self.device,requires_grad=False)
        self.lag_buffer_pos = torch.zeros(self.num_envs,self.cfg.domain_rand.lag_timesteps,self.num_actions,device=self.device,requires_grad=False)
        self.lag_buffer_vel = torch.zeros(self.num_envs,self.cfg.domain_rand.lag_timesteps,self.num_actions,device=self.device,requires_grad=False)
        self.delay_termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg.domain_rand.randomize_imu_offset:
            min_angle, max_angle = self.cfg.domain_rand.randomize_imu_offset_range

            min_angle_rad = math.radians(min_angle)
            max_angle_rad = math.radians(max_angle)

            pitch = torch.rand(self.num_envs, device=self.device) * (max_angle_rad - min_angle_rad) + min_angle_rad
            roll = torch.rand(self.num_envs, device=self.device) * (max_angle_rad - min_angle_rad) + min_angle_rad

            pitch_quat = torch.stack(
                [torch.zeros_like(pitch), torch.sin(pitch / 2), torch.zeros_like(pitch), torch.cos(pitch / 2)], dim=-1)
            roll_quat = torch.stack(
                [torch.sin(roll / 2), torch.zeros_like(roll), torch.zeros_like(roll), torch.cos(roll / 2)], dim=-1)

            self.random_imu_offset = quat_mul(pitch_quat, roll_quat)
        else:
            self.random_imu_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,1)
        
        self.avg_contact_forces = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.env_step_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.sim_time_envs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        #步态相关
        self.gaits = torch.zeros(
            self.num_envs,
            self.cfg.gait.num_gait_params,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs_sin = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs_cos = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # 在 __init__ 或 reset 中加上：
        self.last_contacts_nofly = torch.zeros(
            self.num_envs, len(self.feet_indices),
            device=self.device, dtype=torch.bool
        )

        # -1 表示“还没有发生过起腿事件”
        self.last_swing_leg = torch.full(
            (self.num_envs,), -1,
            device=self.device, dtype=torch.int64
        )

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in ["left_leg_4", "right_leg_4"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.attach_camera(i, env_handle, actor_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float)
        else:
            friction_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].friction
            self.friction_coeffs_tensor = friction_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs_tensor = self.restitution_coeffs.to(self.device).to(torch.float)
        else:
            restitution_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].restitution
            self.restitution_coeffs_tensor = restitution_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [random.randint(0,self.cfg.domain_rand.lag_timesteps-1) for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False
        else:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [self.cfg.domain_rand.lag_timesteps-1 for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        actions = actions.to(self.device)

        self.pre_physics_step()

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.torques = self._scale_and_process_actions(self.torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        base_height_mean = base_height.mean()
        foot_height_mean = self.foot_heights.mean()
        self.env_step_counter += 1
        sim_time_envs = self.env_step_counter * self.cfg.control.decimation * self.cfg.sim.dt
        # 判断静止命令（这里用线速度阈值）
        vel_cmds = self.commands[:, :2]
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
         # feedforward 前馈 (例如只对 hip, knee 2 个关节)
        self.sim_time_envs = sim_time_envs
        t = sim_time_envs
        ff_actions  = self.get_feedforward_action(t, freq=1.67, device=self.device)
        hip_knee_indices = [1,2,5,6]
        # === 静止时替换为策略输出 ===
        # actions shape: (num_envs, num_dofs)
        ff_actions_standing = actions[:, hip_knee_indices].clone()  # (num_envs, 4)

        # mask broadcast
        standing_mask = standing.unsqueeze(1)  # (num_envs, 1)

        # 最终 ff_actions
        ff_actions = torch.where(standing_mask, ff_actions_standing, ff_actions)
        self.extras["q_ref"] = ff_actions

        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras,base_height_mean,foot_height_mean

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

    def get_feedforward_action(self, t, freq=1.0, device="cpu"):
        """
        双足机器人的前馈轨迹生成 (简化正弦轨迹)
        t: 当前时间 (s),tensor shape = (num_envs,)
        freq: 步态频率 (Hz)
        return: (num_envs, 4) -> [hip_L, knee_L, hip_R, knee_R]
        """
        # === 参数设定 ===
        hip_amp = 0.10     # rad
        knee_amp = 0.20    # rad

        hip_default = 0.858
        knee_default = -1.755

        hip_default_t  = torch.tensor(hip_default, device=t.device)
        knee_default_t = torch.tensor(knee_default, device=t.device)

        # === 左腿轨迹 (相位 0) ===
        hip_L = torch.max(hip_default_t, hip_default_t + hip_amp * torch.sin(2 * math.pi * freq * t))
        knee_L = torch.min(knee_default_t, knee_default_t + knee_amp * torch.sin(2 * math.pi * freq * t))

        # === 右腿轨迹 (相位差 π) ===
        hip_R = torch.max(hip_default_t, hip_default_t + hip_amp * torch.sin(2 * math.pi * freq * t + math.pi))
        knee_R = torch.min(knee_default_t, knee_default_t + knee_amp * torch.sin(2 * math.pi * freq * t + math.pi))

        # === 拼接结果 ===
        return torch.stack([hip_L, knee_L, hip_R, knee_R], dim=-1)  # shape (num_envs, 4)
    
    def compute_observations(self):
        self.dof_pos[:,[3, 7]]  = 0 

        obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.action_history_buf[:,-1],
                            # self.clock_inputs_sin.view(self.num_envs, 1),
                            # self.clock_inputs_cos.view(self.num_envs, 1),
                            # self.gaits,
                            ),dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),
                               torch.ones(
                                   8) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   8) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               #torch.zeros(4),
                               torch.zeros(self.num_actions),
                            #    torch.ones(6) * noise_level,
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)
        # add imu noise if needed
        if self.cfg.domain_rand.randomize_imu_offset:
            randomized_base_quat = quat_mul(self.random_imu_offset, self.base_quat)
            obs_buf[:, :3] = quat_rotate_inverse(randomized_base_quat, self.root_states[:, 10:13]) * self.obs_scales.ang_vel
            obs_buf[:, 3:6] = quat_rotate_inverse(randomized_base_quat, self.gravity_vec) 

        priv_latent = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            # self.avg_contact_forces,
            # self.friction_coeffs_tensor,
            self.contact_filt.float()-0.5,
            self.randomized_lag_tensor,
            #self.base_ang_vel  * self.obs_scales.ang_vel,
            # self.base_lin_vel * self.obs_scales.lin_vel,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.restitution_coeffs_tensor,
            self.motor_strength, 
            self.kp_factor,
            self.kd_factor), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # update buffer
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)
    
    #------------- Callbacks --------------
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        #self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self.compute_foot_state()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_cost()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_foot_positions[:] = self.foot_positions[:]

        self.avg_contact_forces = self.contact_forces[:, self.feet_indices, :].reshape(self.contact_forces.shape[0], -1)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    #------------- Cameras --------------
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0],config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return 
        # not meet the requirement of update
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return 
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
        
        self.gym.end_access_image_tensors(self.sim)

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        self._step_contact_targets()
        self._generate_des_ee_ref()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            # self.commands[:, 1] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # if self.global_counter % self.cfg.depth.update_interval == 0:
            self.measured_heights = self._get_heights()
            
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def _process_rigid_body_props(self, props, env_id):
    
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
            rand_mass = self.base_add_mass[env_id].cpu().numpy().reshape(1)

        else:
            self.base_mass[:] = props[0].mass
            rand_mass = np.zeros((1, ))

        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                self.base_add_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_add_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                self.base_add_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                self.base_add_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            props[0].com.x += self.base_add_com[env_id, 0]
            props[0].com.y += self.base_add_com[env_id, 1]
            props[0].com.z += self.base_add_com[env_id, 2]
            rand_com = self.base_add_com[env_id].cpu().numpy().reshape(3)

        else:
            rand_com = np.zeros(3)
            
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        
        mass_params = np.concatenate([rand_mass, rand_com])
        return props,mass_params
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        # === 遍历关节 ===
        for i, name in enumerate(self.dof_names):
            # ---- friction ----
            if self.cfg.domain_rand.radnomize_joint_friction:
                # 先找默认值（模糊匹配）
                friction = 0.02
                for key, val in self.cfg.domain_rand.default_joint_friction.items():
                    if key in name:   # 👈 只要名字里包含关键字
                        friction = val
                        break
                # 随机化
                scale = torch.empty(1, device=self.device).uniform_(
                    self.cfg.domain_rand.rand_joint_friction_range[0],
                    self.cfg.domain_rand.rand_joint_friction_range[1]
                ).item()
                friction *= scale
                props["friction"][i] = friction

            # ---- damping ----
            if self.cfg.domain_rand.radnomize_joint_damping:
                damping = 0.02
                for key, val in self.cfg.domain_rand.default_joint_damping.items():
                    if key in name:   # 👈 模糊匹配
                        damping = val
                        break
                scale = torch.empty(1, device=self.device).uniform_(
                    self.cfg.domain_rand.rand_joint_damping_range[0],
                    self.cfg.domain_rand.rand_joint_damping_range[1]
                ).item()
                damping *= scale
                props["damping"][i] = damping

        if env_id==0:
            for i, name in enumerate(self.dof_names):
                print('props["friction"][i]', props["friction"][i])
                print('props["damping"][i]', props["damping"][i])

        return props
    
    def _low_pass_action_filter(self, actions):
        actons_filtered = self.last_actions * 0.2 + actions * 0.8
        return actons_filtered
    
    def _scale_and_process_actions(self, torques):
        # deadband
        deadband = self.cfg.domain_rand.deadband  # e.g. 0.05 (Nm)
        mask = torch.abs(torques) < deadband * self.deadband_factor
        torques = torques * (~mask)
        return torques

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        pos_action[:, [0, 4]] *= self.cfg.control.hip_scale_reduction
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer_pos = torch.cat([self.lag_buffer_pos[:, 1:, :].clone(),pos_action.unsqueeze(1).clone()], dim=1)
            self.lag_buffer_vel = torch.cat([self.lag_buffer_vel[:, 1:, :].clone(),vel_action.unsqueeze(1).clone()], dim=1)
            delayed_action_pos = self.lag_buffer_pos[torch.arange(self.num_envs), self.randomized_lag, :]
            delayed_action_vel = self.lag_buffer_vel[torch.arange(self.num_envs), self.randomized_lag, :]
        else:
            delayed_action_pos = pos_action
            delayed_action_vel = vel_action
        torques = self.kp_factor*self.p_gains * (delayed_action_pos + self.default_dof_pos - self.dof_pos) + self.kd_factor *self.d_gains * (delayed_action_vel - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.motor_strength #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        overturning_buf = (self.projected_gravity[:, 2] > -0.0)
        delay_termination_buf_step = (termination_contact_buf | overturning_buf)
        self.delay_termination_buf += delay_termination_buf_step
        self.reset_buf = (self.delay_termination_buf > self.cfg.env.delay_termination_time_s / self.dt)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_foot_state(self):
        self.foot_positions = self.rigid_body_states[:, self.feet_indices,
                              0:3]
        self.foot_velocities = (
            self.foot_positions - self.last_foot_positions
        ) / self.dt
        
        self.foot_heights = torch.clip(
            (
                self.foot_positions[:, :, 2]
                - self.cfg.asset.foot_radius
                - self._get_foot_heights()), 0, 1)
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = torch.clip(
                rew,
                -self.cfg.rewards.clip_single_reward,
                self.cfg.rewards.clip_single_reward,
            )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        self.rew_buf[:] = torch.clip(
            self.rew_buf[:], -self.cfg.rewards.clip_reward, self.cfg.rewards.clip_reward
        )
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_cost(self):
        self.cost_buf[:] = 0
        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.dt #self.cost_scales[name]
            self.cost_buf[:,i] += cost
            self.cost_episode_sums[name] += cost

    def randomize_hip_joints(self, env_ids):
        """只修改一半环境的两个髋关节角度"""
        if len(env_ids) == 0:
            return
        
        # 只取一半的环境进行修改
        num_to_modify = max(1, len(env_ids) // 2)  # 至少修改1个环境
        env_ids_to_modify = env_ids[:num_to_modify]

        hip_l_idx = 1
        hip_r_idx = 5
        
        # 修改指定环境的髋关节角度
        self.dof_pos[env_ids_to_modify, hip_l_idx] = -0.9  # 左髋关节
        self.dof_pos[env_ids_to_modify, hip_r_idx] = 0.9   # 右髋关节
        
        # 重置速度
        self.dof_vel[env_ids_to_modify] = 0.0
        
        # 更新物理引擎
        env_ids_int32 = env_ids_to_modify.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        # self.randomize_hip_joints(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.delay_termination_buf[env_ids] = 0.
        self.last_contacts_nofly[env_ids] = False
        self.stable_contact_counter[env_ids] = 0.0
        self.env_step_counter[env_ids] = 0
        self.gait_indices[env_ids] = 0
        # 初始化 no_fly 相关状态
        self.last_contacts_nofly[env_ids] = False
        self.last_swing_leg[env_ids] = -1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0
        self.lag_buffer_pos[env_ids,:,:] = 0
        self.lag_buffer_vel[env_ids,:,:] = 0
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs,_,_, _, _,_,_,_,= self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # random ori
        # self.root_states[env_ids, 3:7] = random_quat(torch_rand_float(0, 1, (len(env_ids), 4), device=self.device))

        # Use a fixed initial rotation
        self.root_states[env_ids, 3:7] = self.base_init_state[3:7]        # random height
        
        self.root_states[env_ids, 2:3] += torch_rand_float(0, 0.2, (len(env_ids), 1), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids,:] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        
    def _prepare_cost_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale==0:
                self.cost_scales.pop(key) 
            # else:
            #     self.cost_scales[key] *= self.dt

        self.cost_functions = []
        self.cost_names = []
        self.cost_k_values = []
        self.cost_d_values_tensor = []

        for name,scale in self.cost_scales.items():
            self.cost_names.append(name)
            name = '_cost_' + name
            print('cost name:',name)
            print('cost k value:',scale)
            self.cost_functions.append(getattr(self, name))
            self.cost_k_values.append(float(scale))

        for name,value in self.cost_d_values.items():
            print('cost name:',name)
            print('cost d value:',value)
            self.cost_d_values_tensor.append(float(value))

        self.cost_k_values = torch.FloatTensor(self.cost_k_values).view(1,-1).to(self.device)
        self.cost_d_values_tensor = torch.FloatTensor(self.cost_d_values_tensor).view(1,1,-1).to(self.device)

        self.cost_episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                  for name in self.cost_scales.keys()}

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        print("dt is", self.dt)
        print("sim_params.dt is", self.sim_params.dt)
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.cost_scales = class_to_dict(self.cfg.costs.scales)
        self.cost_d_values = class_to_dict(self.cfg.costs.d_values)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        # global counter 是否该类似这个
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.gaits_ranges = class_to_dict(self.cfg.gait.ranges)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        # draw depth image with window created by cv2
        if self.cfg.depth.use_camera:
            window_name = "Depth Image"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cup().numpy() + 0.5)
            cv2.waitKey(1) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    #------------ curriculum ----------------
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbancFe_range[0], self.cfg.domain_rand.disturbancFe_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _resample_commands(self, env_ids):
        # """ Randommly select commands of some environments
        # Args:
        #     env_ids (List[int]): Environments ids for which new commands are needed
        # """
        # lin_vel_x
        zero_mask_x = torch.rand(len(env_ids), device=self.device) < 0.1
        rand_x = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        rand_x[zero_mask_x] = 0.0
        self.commands[env_ids, 0] = rand_x

        # # lin_vel_y
        zero_mask_y = torch.rand(len(env_ids), device=self.device) < 0.1
        rand_y = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        rand_y[zero_mask_y] = 0.0
        self.commands[env_ids, 1] = rand_y
        # self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_foot_heights(self):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.foot_positions[:, :, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights

    def _step_contact_targets(self):
            frequencies = self.gaits[:, 0]
            offsets = self.gaits[:, 1]
            durations = torch.cat(
                [
                    self.gaits[:, 2].view(self.num_envs, 1),
                    self.gaits[:, 2].view(self.num_envs, 1),
                ],
                dim=1,
            )
            self.gait_indices = torch.remainder(
                self.gait_indices + self.dt * frequencies, 1.0
            )

            self.clock_inputs_sin = torch.sin(2 * np.pi * self.gait_indices)
            self.clock_inputs_cos = torch.cos(2 * np.pi * self.gait_indices)
            # self.doubletime_clock_inputs_sin = torch.sin(4 * np.pi * foot_indices)
            # self.halftime_clock_inputs_sin = torch.sin(np.pi * foot_indices)

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

            foot_indices = torch.remainder(
                torch.cat(
                    [
                        self.gait_indices.view(self.num_envs, 1),
                        (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                    ],
                    dim=1,
                ),
                1.0,
            )
            stance_idxs = foot_indices < durations
            swing_idxs = foot_indices > durations

            foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
                0.5 / durations[stance_idxs]
            )
            foot_indices[swing_idxs] = 0.5 + (
                torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
            ) * (0.5 / (1 - durations[swing_idxs]))

            self.desired_contact_states = smoothing_cdf_start(foot_indices) * (
                1 - smoothing_cdf_start(foot_indices - 0.5)
            ) + smoothing_cdf_start(foot_indices - 1) * (
                1 - smoothing_cdf_start(foot_indices - 1.5)
            )

    def _generate_des_ee_ref(self):
        frequencies = self.gaits[:, 0]
        mask_0 = (self.gait_indices < 0.25) & (self.gait_indices >= 0.0)  # lift up
        mask_1 = (self.gait_indices < 0.5) & (self.gait_indices >= 0.25)  # touch down
        mask_2 = (self.gait_indices < 0.75) & (self.gait_indices >= 0.5)  # lift up
        mask_3 = (self.gait_indices <= 1.0) & (self.gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.cfg.gait.touch_down_vel
        swing_end_vel[mask_3] = self.cfg.gait.touch_down_vel

        # generate desire foot z trajectory
        swing_height = self.gaits[:, 3]
        # self.des_foot_height = 0.5 * swing_height * (1 - torch.cos(4 * np.pi * self.gait_indices))
        # self.des_foot_velocity_z = 2 * np.pi * swing_height * frequencies * torch.sin(
        #     4 * np.pi * self.gait_indices)

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(self.gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(self.gait_indices / frequencies)

    def _resample_gaits(self, env_ids):
        if len(env_ids) == 0:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 4
        # self.gaits[env_ids, 1] = (self.gaits[env_ids, 1] * parts).round() / parts
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 2
        # self.gaits[env_ids, 2] = (self.gaits[env_ids, 2] * parts).round() / parts

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def get_feedforward_action(self, t, freq=1.0, device="cpu"): 
        """ 双足机器人的前馈轨迹生成 (简化正弦轨迹) 
        t: 当前时间 (s),tensor shape = (num_envs,) 
        freq: 步态频率 (Hz) return: (num_envs, 4) -> [hip_L, knee_L, hip_R, knee_R] """ 
        # === 参数设定 === 
        hip_amp = 0.10 # rad 
        knee_amp = 0.20 # rad 

        hip_default = 0.0
        knee_default = -0.0

        hip_default_t = torch.tensor(hip_default, device=t.device) 
        knee_default_t = torch.tensor(knee_default, device=t.device) 

        # === 左腿轨迹 (相位 0) === 
        hip_L = torch.max(hip_default_t, hip_default_t + hip_amp * torch.sin(2 * math.pi * freq * t)) 
        knee_L = torch.min(knee_default_t, knee_default_t + knee_amp * torch.sin(2 * math.pi * freq * t)) 
        # === 右腿轨迹 (相位差 π) === 
        hip_R = torch.max(hip_default_t, hip_default_t + hip_amp * torch.sin(2 * math.pi * freq * t + math.pi)) 
        knee_R = torch.min(knee_default_t, knee_default_t + knee_amp * torch.sin(2 * math.pi * freq * t + math.pi)) 
        # === 拼接结果 === 
        return torch.stack([hip_L, knee_L, hip_R, knee_R], dim=-1) # shape (num_envs, 4) 
    
    def _reward_reference_trajectory(self): 
        # === 当前仿真时间 (num_envs,) === 
        t = self.sim_time_envs # === 获取参考轨迹 === 
        q_ref = self.get_feedforward_action(t, freq=1.67, device=self.device) # [num_envs, 4] 
        # === 实际 hip/knee 关节角度 === 
        hip_knee_indices = [1, 2, 5, 6] # 对应 hip_L, knee_L, hip_R, knee_R 
        q = self.dof_pos[:, hip_knee_indices] 
        # === 位置跟踪奖励 === 
        diff = q - q_ref 
        r_pos = torch.exp(-torch.sum(diff**2, dim=-1) / 0.05) 
        # 判断静止命令（这里用线速度阈值） 
        vel_cmds = self.commands[:, :2] 
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1) 
        moving = ~standing 
        return r_pos*moving
    
    #步态相关奖励
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                    )
                )

        # 判断静止命令（这里用线速度阈值）
        vel_cmds = self.commands[:, :2]
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        moving = ~standing
        return torch.where(moving, reward / len(self.feet_indices), 0)
        # return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                    )
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (1 - torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma)
                )
        # 判断静止命令（这里用线速度阈值）
        vel_cmds = self.commands[:, :2]
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        moving = ~standing
        return torch.where(moving, reward / len(self.feet_indices), 0)
        # return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_height(self):
        foot_heights = self.foot_heights
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_height"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma)
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma)
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma))
        # 判断静止命令（这里用线速度阈值）
        vel_cmds = self.commands[:, :2]
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        moving = ~standing
        return torch.where(moving, reward / len(self.feet_indices), 0)
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_powers(self):
        # Penalize torques
        #return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        return  torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        #rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)
        #rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        vel_cmds = self.commands[:,:3]  # (N, 3)
        # 判断是否每个指令都小于等于 0.1（包括负方向）
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        default_dof_pos_clone = self.default_dof_pos.clone()
        default_dof_pos_clone[:, [0,1,2,4,5,6]] = torch.tensor([0.0, 0.225, 0.707, 0.0, -0.225, -0.707], 
                                                      dtype=default_dof_pos_clone.dtype, 
                                                      device=default_dof_pos_clone.device)
        return torch.sum(torch.abs(self.dof_pos[:, [0,1,2,4,5,6]] - default_dof_pos_clone[:, [0,1,2,4,5,6]]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_feet_distance(self):
        vel_cmds = self.commands[:,:3]  # (N, 3)
        # 判断是否每个指令都小于等于 0.1（包括负方向）
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        feet_distance = torch.abs(torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    # def _reward_no_fly(self):
    #     contact_now = self.contact_forces[:, self.feet_indices, 2] > 0.5
    #     num_contacts_now = torch.sum(contact_now.float(), dim=1)
        
    #     # 步态和接触信息
    #     desired_contact = self.desired_contact_states
    #     left_desired, right_desired = desired_contact[:, 0], desired_contact[:, 1]
    #     left_actual, right_actual = contact_now[:, 0], contact_now[:, 1]
        
    #     # 移动状态
    #     vel_cmds = self.commands[:, :2]
    #     actual_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
    #     standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1) & (actual_vel < 0.2)
    #     moving = ~standing

    #     reward = torch.ones_like(moving, dtype=torch.float) * 0.5

    #     # === 基础状态奖励 ===
    #     reward[(num_contacts_now == 2) & standing] = 1.0
    #     reward[(num_contacts_now == 1) & moving] = 1.0
    #     reward[(num_contacts_now == 0) & moving] = -2.0
    #     reward[(num_contacts_now == 2) & moving] = -1.5

    #     # === 核心：精确的步态匹配检测 ===
    #     if moving.any():
    #         # 1. 接触状态匹配度（使用实际接触状态）
    #         contact_match = torch.abs(contact_now.float() - desired_contact)
    #         match_quality = 1.0 - torch.mean(contact_match, dim=1)
            
    #         reward[moving] += match_quality[moving] * 0.6  # 匹配度奖励
            
    #         # 2. 特定状态检测
    #         # 期望单脚支撑但实际双脚接触
    #         expected_single = (left_desired + right_desired) < 1.5
    #         wrong_double_contact = expected_single & (num_contacts_now == 2)
    #         reward[wrong_double_contact & moving] -= 0.7
            
    #         # 期望双脚支撑但实际单脚接触
    #         expected_double = (left_desired + right_desired) > 1.5
    #         wrong_single_contact = expected_double & (num_contacts_now == 1)
    #         reward[wrong_single_contact & moving] -= 0.5
            
    #         # 3. 左右脚精确匹配
    #         left_match = (left_desired > 0.5) == left_actual
    #         right_match = (right_desired > 0.5) == right_actual
            
    #         # 完美匹配奖励
    #         perfect_match = left_match & right_match
    #         reward[perfect_match & moving] += 0.4
            
    #         # 严重不匹配惩罚
    #         severe_mismatch = ~left_match & ~right_match
    #         reward[severe_mismatch & moving] -= 0.6

    #     self.last_contacts_nofly = contact_now.clone()
    #     return reward
    
    def _reward_no_fly(self):
        #wheel
        # 当前帧接触情况
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 0.1
        num_contacts_now = torch.sum(contact_now.float(), dim=1)

        # 判断静止命令（这里用线速度阈值）
        vel_cmds = self.commands[:, :2]
        standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1)
        moving = ~standing

        # 静止逻辑：需要两条腿同时接触
        correct_standing = (num_contacts_now == 2)

        # 运动逻辑：只允许一条腿接触
        #correct_moving = (num_contacts_now == 1)
        correct_moving = (num_contacts_now > 0)

        # 初始化奖励
        reward = torch.zeros_like(moving, dtype=torch.float)

        # 静止奖励
        reward[standing] = correct_standing[standing].float()

        # 运动奖励
        reward[moving] = correct_moving[moving].float()

        # 可选：保存接触信息用于其他奖励
        self.last_contacts_nofly = contact_now.clone()
        # 当前帧接触情况
        #biped
        # contact_now = self.contact_forces[:, self.feet_indices, 2] > 0.5
        # num_contacts_now = torch.sum(contact_now.float(), dim=1)

        # # 判断是否在移动（结合命令和实际速度）
        # vel_cmds = self.commands[:, :2]
        # actual_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        
        # # 静止条件：命令速度小且实际速度小
        # # standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1) & (actual_vel < 0.2)
        # standing = torch.all(torch.abs(vel_cmds) < 0.1, dim=1) 
        # moving = ~standing

        # # 静止逻辑：需要两条腿同时接触
        # correct_standing = (num_contacts_now == 2)
        # wrong_standing = (num_contacts_now != 2) & standing

        # # 运动逻辑：只允许一条腿接触
        # correct_moving = (num_contacts_now == 1)
        # no_contact_moving = (num_contacts_now == 0) & moving  # 飞行惩罚
        # both_contact_moving = (num_contacts_now == 2) & moving  # 双足接触惩罚

        # # 初始化奖励
        # reward = torch.ones_like(moving, dtype=torch.float) * 0.5  # 基础奖励

        # # 静止状态奖励/惩罚
        # reward[correct_standing & standing] = 1.0  # 正确静止：最大奖励
        # reward[wrong_standing] = -1.0  # 错误静止：惩罚

        # # 运动状态奖励/惩罚
        # reward[correct_moving & moving] = 1.0  # 正确运动：最大奖励
        # reward[no_contact_moving] = -2.0  # 飞行：严重惩罚
        # # reward[both_contact_moving] = -1.5  # 双足接触：中等惩罚
        # reward[both_contact_moving] = 1.0  #

        # # 可选：添加连续正确步态的额外奖励
        # if hasattr(self, 'last_moving_correct'):
        #     # 连续正确步态奖励
        #     consecutive_correct = correct_moving & moving & self.last_moving_correct
        #     reward[consecutive_correct] += 0.3  # 连续正确额外奖励
        # self.last_moving_correct = correct_moving & moving

        # # 保存接触信息
        # self.last_contacts_nofly = contact_now.clone()

        return reward
    
    def _reward_hip_symmetry(self):
        # 假设 0 号和 4 号 DOF 是左右髋关节
        left_hip = self.dof_pos[:, 0].clone()
        right_hip = self.dof_pos[:, 4].clone()

        # 对称性 = 两个角度接近，差值越小越好
        hip_diff = left_hip + right_hip  # 如果左右对称，差值应接近 0（假设一条腿正方向，另一条负方向）
        
        # 奖励 = 差值的负平方
        reward = torch.square(hip_diff)

        return reward
    
    def _reward_hip_pos(self):
        default_dof_pos_clone = self.default_dof_pos.clone()
        default_dof_pos_clone[:, [0, 4]] = 0.0
        return torch.sum(torch.square(self.dof_pos[:, [0, 4]] - default_dof_pos_clone[:, [0, 4]]), dim=1)
    
    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        # return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
        return torch.abs(height_error) / len(self.feet_indices) 
    
    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        # return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)
        return torch.abs(leg_symmetry_err)
    
    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2
    
    #------------ cost functions----------------
    def _cost_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        # reward = torch.abs(foot_x_position_err)
        reward = torch.abs(foot_x_position_err - foot_x_position_err)
        return reward
    
    def _cost_torque_limit(self):
        # constaint torque over limit
        return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
    
    def _cost_pos_limit(self):
        upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        return out_limit
   
    def _cost_dof_vel_limits(self):
        return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
    
    def _cost_vel_smoothness(self):
        return torch.mean(torch.max(torch.zeros_like(self.dof_vel),torch.abs(self.dof_vel) - (self.dof_vel_limits/2.)),dim=1)
    
    def _cost_acc_smoothness(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        acc_limit = self.dof_vel_limits/(2.*self.dt)
        return torch.mean(torch.max(torch.zeros_like(acc),torch.abs(acc) - acc_limit),dim=1)
    
    def _cost_collision(self):
        return  1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) > 0.0)
    
    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)
        # return torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
    
    def _cost_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)
    
    def _cost_stumble_up(self):
        # Penalize feet hitting vertical surfaces
        return torch.clamp(-self.projected_gravity[:,2],0,1)*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)
    
    def _cost_feet_contact_forces_up(self):
        # penalize high contact forces
        return torch.clamp(-self.projected_gravity[:,2],0,1)*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)

    def _cost_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _cost_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return 1.*(rew_airTime < 0.0)
    
    def _cost_ang_vel_xy(self):
        ang_vel_xy = 0.01*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return ang_vel_xy
    
    def _cost_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _cost_torques(self):
        # Penalize torques
        torque_squres = 0.0001*torch.sum(torch.square(self.torques),dim=1)
        return torque_squres
    
    def _cost_action_rate(self):
        action_rate = 0.01*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate

    def _cost_action_smoothness(self):
        return  torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)
    
    def _cost_walking_style(self):
        # number of contact must greater than 2 at each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        return 1.*(torch.sum(1.*contact_filt,dim=-1) < 3.)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _cost_hip_pos(self):
        default_dof_pos_clone = self.default_dof_pos.clone()
        # default_dof_pos_clone[:, 0] = 0.0
        # default_dof_pos_clone[:, 4] = -0.0
        default_dof_pos_clone[:, [0, 4]] = self.dof_pos[:, [0, 4]].clone()
        return torch.sum(torch.square(self.dof_pos[:, [0, 4]] - default_dof_pos_clone[:, [0, 4]]), dim=1)
    
    def _cost_feet_height(self):
        # Reward high steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        foot_heights_cost = torch.sum(torch.square(self.dof_pos[:,[2,5,8,11]] - (-2.0)) * (~contact_filt),dim=1)
 
        return foot_heights_cost
    
    def _cost_contact_force_xy(self):
        contact_xy_force_norm = torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :2],dim=-1),dim=-1)
        return contact_xy_force_norm

    def _cost_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _cost_default_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _cost_feet_distance(self):
        feet_distance = torch.abs(torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward
    
    def _cost_powers(self):
        # Penalize torques
        #return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)