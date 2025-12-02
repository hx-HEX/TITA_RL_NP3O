import cv2
import os
import pygame  # 替换keyboard库
import numpy as np
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from isaacgym import gymapi
import torch
from utils import get_args, export_policy_as_jit, task_registry, Logger
from configs import *
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
from global_config import ROOT_DIR
from configs.tita_wheel_constraint_config import TitaConstraintWheelCfg, TitaConstraintWheelCfgPPO
from configs.tita_feet_constraint_config import TitaConstraintFeetCfg,TitaConstraintFeetCfgPPO
from configs.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from configs.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from configs.tron1_wheel_constraint_config import Tron1ConstraintWheelCfg, Tron1ConstraintWheelCfgPPO
from configs.tron1_feet_constraint_config import Tron1ConstraintFeetCfg, Tron1ConstraintFeetCfgPPO
from envs.no_constrains_legged_robot import Tita
from envs import *
from export_policy_as_onnx import *
import argparse
from configs.tita_fusion_constraint_config import TitaFusionCfg, TitaFusionCfgPPO
from utils.helpers import hard_phase_schedualer, partial_checkpoint_load,load_expert_from_file

# 设置是否录制帧
RECORD_FRAMES = False

def play_on_constraint_policy_runner(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    # env_cfg.terrain.terrain_length = 5
    # env_cfg.terrain.terrain_width = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_imu_offset = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.control.use_filter = True
    env_cfg.commands.heading_command = False
    env_cfg.domain_rand.radnomize_joint_friction = False
    env_cfg.domain_rand.radnomize_joint_damping = False
    env_cfg.domain_rand.randomize_deadband = False
    env_cfg.env.episode_length_s = 500
    env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2]
    env_cfg.init_state.pos = [-0.0,0.0, 0.8]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runnner_cfg = train_cfg.runner

    if runnner_cfg.policy_class_name == "FusionPolicyWithCritic":
        training_type = "fusion"
        print("training_type is FusionPolicyWithCritic")
    else:
        training_type = "rl"

    if training_type == "rl":
        # load policy partial_checkpoint_load
        policy_cfg_dict = class_to_dict(train_cfg.policy)
        runner_cfg_dict = class_to_dict(train_cfg.runner)
        actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
        policy = actor_critic_class(env.cfg.env.n_proprio,
                                env.cfg.env.n_scan,
                                env.num_obs,
                                env.cfg.env.n_priv_latent,
                                env.cfg.env.history_len,
                                env.num_actions,
                                **policy_cfg_dict)
        
    else:
        wheel_cfg = train_cfg.runner.experts["wheel"]
        biped_cfg = train_cfg.runner.experts["biped"]
        print("training_type is FusionPolicyWithCritic")
        expert_wheel = load_expert_from_file(wheel_cfg["cfg_path"], wheel_cfg["ckpt_path"]) 
        expert_biped = load_expert_from_file(biped_cfg["cfg_path"], biped_cfg["ckpt_path"])
        policy_cfg_dict = class_to_dict(train_cfg.policy)
        runner_cfg_dict = class_to_dict(train_cfg.runner)
        actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
        policy = actor_critic_class(expert_wheel,
                                    expert_biped,
                                    env.cfg.env.n_proprio,
                                    env.cfg.env.n_scan,
                                    env.cfg.env.n_priv_latent,
                                    env.cfg.env.history_len,
                                    16,
                                    env.num_actions,
                                    **policy_cfg_dict)

    # print(policy)
    
    # 加载模型权重
    model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/tron1_feet_constraint/Nov10_14-47-36_tron1_feet_barlowtwins/model_1200.pt'))
    policy.load_state_dict(model_dict['model_state_dict'])
    policy = policy.to(env.device)
    policy.save_torch_jit_policy('model.pt', env.device)
    
    # 把模型移动到 CPU
    policy_cpu = policy.to("cpu")
    policy_cpu.save_torch_jit_policy('model_cpu.pt', device='cpu')
    policy = policy.to(env.device)

    # ====== 使用pygame键盘控制部分 ======
    pygame.init()
    # 创建一个小窗口用于接收键盘输入
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Keyboard Control")
    
    # 键盘映射配置
    KEY_MAPPING = {
        pygame.K_UP: [0.5, 0, 0, 0],      # 前进
        pygame.K_DOWN: [-1.0, 0, 0, 0],   # 后退
        pygame.K_LEFT: [0, 0.0, 0.6, 0],    # 左移
        pygame.K_RIGHT: [0, -0.0, -0.6, 0],  # 右移
        pygame.K_w: [0, 0, 0.5, 0],       # 升高身体
        pygame.K_s: [0, 0, -0.5, 0],      # 降低身体
        pygame.K_a: [0, 0.5, 0, 0.5],       # 左转
        pygame.K_d: [0, -0.5, 0, -0.5],      # 右转
        pygame.K_SPACE: [0, 0, 0, 0]      # 停止
    }
    
    current_command = np.zeros(4)
    # ====== pygame键盘控制部分结束 ======
    
    # set rgba camera sensor for debug and double check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0
    video_duration = 200
    # video_duration = 25
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    # 新增：数据存储容器
    all_data = {
        'pri_latents': [],
        'hist_latents': [],
        'linear_vel': [],  # 线速度 [vx, vy, vz]
        'obs': [],
        'actions': [],
        'timesteps': []
    }

    print("按ESC键退出控制")
    print("控制键: ↑↓←→移动, WASD高度/转向, SPACE停止")
    
    try:
        for i in range(num_frames):
            # ====== pygame键盘事件处理 ======
            command_delta = np.zeros(4)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # 获取当前按键状态
            keys = pygame.key.get_pressed()
            for key_code, command in KEY_MAPPING.items():
                if keys[key_code]:
                    command_delta += np.array(command)
            
            # 平滑处理命令
            current_command = 0.8 * current_command + 0.2 * command_delta
            
            # 更新环境命令
            env.commands[:, 0] = current_command[0]  # lin_vel_x
            env.commands[:, 1] = current_command[1]  # lin_vel_y
            env.commands[:, 2] = current_command[2]  # body_height
            env.commands[:, 3] = current_command[3]  # ang_vel_yaw
            # env.commands[:, 0] = 1.0 # lin_vel_x
            # env.commands[:, 1] = 0.0 # lin_vel_y
            # env.commands[:, 2] = 0.0  # body_height
            # env.commands[:, 3] = 0.0  # ang_vel_yaw
            # 检查退出键
            if keys[pygame.K_ESCAPE]:
                print("ESC键按下,退出程序")
                break
            # ====== 键盘事件处理结束 ======
            
            # 更新统计变量
            action_rate += torch.sum(torch.abs(env.last_actions - env.actions), dim=1)
            z_vel += torch.square(env.base_lin_vel[:, 2])
            xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

            # 使用策略生成动作
            actions = policy.act_teacher(obs)
            if training_type == "fusion":
                print("weight",policy.last_attention_weights)
            # actions, pri_latents, hist_latents= get_latents_during_inference(policy,obs)
            # linear_velocity = env.base_lin_vel[0].cpu().numpy() 
            # # 保存数据
            # all_data['pri_latents'].append(pri_latents[0].cpu().numpy())  # [1, 3]
            # all_data['hist_latents'].append(hist_latents[0].cpu().numpy())  # [1, latent_dim]
            # all_data['linear_vel'].append(linear_velocity)  # [3,]
            # all_data['obs'].append(obs.cpu().numpy())
            # all_data['actions'].append(actions.cpu().numpy())
            # all_data['timesteps'].append(i * env.dt)  # 记录时间戳

            obs, privileged_obs, rewards, costs, dones, infos, base_height,foot_height_mean= env.step(actions)
            # obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)
            # print("actions[:,3]*4",actions[:,3]*4)
            # print("obs[:,20]/0.05",obs[:,20]/0.05)
            # print("actions[:,7]*4",actions[:,7]*4)
            # print("obs[:,24]/0.05",obs[:,24]/0.05)
            # print("pri_latents[0]",pri_latents[0])
            print("foot_height_mean",foot_height_mean)
      
            # 渲染
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            # print("base_height rate:", base_height)
            
            if RECORD_FRAMES:
                img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512,512,4))[:,:,:3]
                if video is None:
                    video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
                video.write(img)
                img_idx += 1 
                
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理pygame
        pygame.quit()
        
        # 释放视频资源
        if RECORD_FRAMES and video is not None:
            video.release()
        
        # 输出统计信息
        print("action rate:", action_rate/num_frames)
        print("z vel:", z_vel/num_frames)
        print("xy_vel:", xy_vel/num_frames)

        # 合并数据并保存
        # np.savez('sim_data_rough_slope.npz',
        #          pri_latents=np.vstack(all_data['pri_latents']),
        #          hist_latents=np.vstack(all_data['hist_latents']),
        #          linear_vel=np.vstack(all_data['linear_vel']),
        #          obs=np.vstack(all_data['obs']),
        #          actions=np.vstack(all_data['actions']),
        #          timesteps=np.array(all_data['timesteps']))
        
def get_latents_during_inference(policy, obs):
    """
    在不修改模型的情况下获取latents
    返回: (actions, latents)
    """
    captured = []
    
    # 定义钩子
    def hook(module, inp, out):
        captured.append(out.detach().clone())
    
    # 注册钩子
    handle = policy.actor_teacher_backbone.mlp_encoder.register_forward_hook(hook)
    
    # 执行推理
    with torch.no_grad():
        actions = policy.act_teacher(obs)
        pri_latents = captured[0][:, :3]  # 假设latent_dim=16，提取前16维
        hist_latents = captured[0][:, 3:]
    
    # 清理钩子
    handle.remove()
    
    return actions, pri_latents, hist_latents

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # Register tasks
    task_registry.register("tita_flat", Tita, TitaFlatCfg(), TitaFlatCfgPPO())
    task_registry.register("tita_rough", Tita, TitaRoughCfg(), TitaRoughCfgPPO())
    task_registry.register("tita_constraint", LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())
    task_registry.register("tita_wheel_constraint",LeggedRobot,TitaConstraintWheelCfg(),TitaConstraintWheelCfgPPO())
    task_registry.register("tita_feet_constraint",LeggedRobot,TitaConstraintFeetCfg(),TitaConstraintFeetCfgPPO())
    task_registry.register("tita_fusion_constraint", LeggedRobot, TitaFusionCfg(), TitaFusionCfgPPO())
    task_registry.register("tron1_wheel_constraint", Tron1Robot, Tron1ConstraintWheelCfg(), Tron1ConstraintWheelCfgPPO())
    task_registry.register("tron1_feet_constraint", Tron1FeetRobot, Tron1ConstraintFeetCfg(), Tron1ConstraintFeetCfgPPO())
    args = get_args()
    play_on_constraint_policy_runner(args)