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
from configs.tron1_unified_constraint_config import Tron1ConstraintUnifiedCfg, Tron1ConstraintUnifiedCfgPPO
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
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_imu_offset = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.control.use_filter = True
    env_cfg.commands.heading_command = False
    env_cfg.env.episode_length_s = 500
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.init_state.pos = [-0.0, 0.0, 0.8]

    # -------------------------------------------
    # prepare environment
    # -------------------------------------------
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runnner_cfg = train_cfg.runner

    # load policy
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

    # 加载模型权重
    model_dict = torch.load(os.path.join(ROOT_DIR,
                    'logs/tron1_unified_constraint/Dec01_23-38-10_tron1_unified_barlowtwins/model_30000.pt'))
    policy.load_state_dict(model_dict['model_state_dict'])
    policy = policy.to(env.device)

    # -------------------------------------------
    # pygame keyboard
    # -------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((120, 120))
    pygame.display.set_caption("Keyboard Control")

    # === command indices ===
    CMD_LIN_X = 0
    CMD_LIN_Y = 1
    CMD_YAW = 2
    CMD_HEIGHT = 4
    CMD_GAIT = 5
    CMD_JUMP = 6
    CMD_SPIN = 7

    # default height
    DEFAULT_HEIGHT = 0.7
    height_cmd_min, height_cmd_max = 0.65, 0.80

    # === continuous command state ===
    vel_cmd = np.zeros(3)
    height_cmd = DEFAULT_HEIGHT

    # === binary command state ===
    gait_mode = 0
    jump_flag = 0
    spin_flag = 0

    # === key toggle states ===
    key_toggle_state = {k: False for k in [pygame.K_g, pygame.K_j, pygame.K_h]}

    # Steps
    LIN_VEL_STEP = 0.05
    YAW_STEP = 0.05
    HEIGHT_STEP = 0.01

    print("\n======== Keyboard Control ========")
    print("↑↓：前后速度")
    print("←→：左右速度")
    print("A/D：左转 / 右转")
    print("W/S：升高 / 降低 body_height")
    print("G：切换 gait_mode")
    print("J：切换 jump_flag")
    print("H：切换 spin_flag")
    print("SPACE：恢复默认命令")
    print("ESC：退出")
    print("===================================\n")

    # -------- camera ----------
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0, 0, 1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0],
                                                      env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(
        cam_handle, env.envs[0], body_handle, camera_local_transform,
        gymapi.FOLLOW_TRANSFORM)

    num_frames = int(200 / env.dt)
    last_print_cmd = None

    try:
        for i in range(num_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                print("ESC pressed, exit.")
                break

            # ============================================================
            # SPACE → Reset all commands
            # ============================================================
            if keys[pygame.K_SPACE]:
                vel_cmd[:] = 0
                height_cmd = DEFAULT_HEIGHT
                gait_mode = 0
                jump_flag = 0
                spin_flag = 0

            # ============================================================
            # Continuous Commands
            # ============================================================
            if keys[pygame.K_UP]:
                vel_cmd[0] += LIN_VEL_STEP
            if keys[pygame.K_DOWN]:
                vel_cmd[0] -= LIN_VEL_STEP

            if keys[pygame.K_LEFT]:
                vel_cmd[1] += LIN_VEL_STEP
            if keys[pygame.K_RIGHT]:
                vel_cmd[1] -= LIN_VEL_STEP

            if keys[pygame.K_a]:
                vel_cmd[2] += YAW_STEP
            if keys[pygame.K_d]:
                vel_cmd[2] -= YAW_STEP

            if keys[pygame.K_w]:
                height_cmd += HEIGHT_STEP
            if keys[pygame.K_s]:
                height_cmd -= HEIGHT_STEP

            # clamp
            vel_cmd[0] = np.clip(vel_cmd[0], -1.0, 1.0)
            vel_cmd[1] = np.clip(vel_cmd[1], -1.0, 1.0)
            height_cmd = np.clip(height_cmd, height_cmd_min, height_cmd_max)

            # ============================================================
            # Binary Commands (with edge detection)
            # ============================================================
            if keys[pygame.K_g]:
                if not key_toggle_state[pygame.K_g]:
                    gait_mode = 1 - gait_mode
                    key_toggle_state[pygame.K_g] = True
            else:
                key_toggle_state[pygame.K_g] = False

            if keys[pygame.K_j]:
                if not key_toggle_state[pygame.K_j]:
                    jump_flag = 1 - jump_flag
                    key_toggle_state[pygame.K_j] = True
            else:
                key_toggle_state[pygame.K_j] = False

            if keys[pygame.K_h]:
                if not key_toggle_state[pygame.K_h]:
                    spin_flag = 1 - spin_flag
                    key_toggle_state[pygame.K_h] = True
            else:
                key_toggle_state[pygame.K_h] = False

            if spin_flag == 0:
                vel_cmd[2] = np.clip(vel_cmd[2], -1, 1)
            else:
                vel_cmd[2] = np.clip(vel_cmd[2], -20, 20)
            # ============================================================
            # 写入 env.commands
            # ============================================================
            env.commands[:, CMD_LIN_X] = vel_cmd[0]
            env.commands[:, CMD_LIN_Y] = vel_cmd[1]
            env.commands[:, CMD_YAW] = vel_cmd[2]
            env.commands[:, CMD_HEIGHT] = height_cmd
            env.commands[:, CMD_GAIT] = gait_mode
            env.commands[:, CMD_JUMP] = jump_flag
            env.commands[:, CMD_SPIN] = spin_flag

            # ============================================================
            # 实时显示 command（仅在变化时打印）
            # ============================================================
            current_cmd = (
                round(vel_cmd[0], 3),
                round(vel_cmd[1], 3),
                round(vel_cmd[2], 3),
                round(height_cmd, 3),
                gait_mode,
                jump_flag,
                spin_flag,
            )

            if current_cmd != last_print_cmd:
                print(f"CMD: vx={current_cmd[0]}  vy={current_cmd[1]}  yaw={current_cmd[2]}  "
                      f"h={current_cmd[3]}  gait={gait_mode}  jump={jump_flag}  spin={spin_flag}")
                last_print_cmd = current_cmd

            # ============================================================
            # policy
            # ============================================================
            actions = policy.act_teacher(obs)
            obs, privileged_obs, rewards, costs, dones, infos, base_height, foot_height_mean = env.step(actions,1)

            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        pygame.quit()


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
    task_registry.register("tron1_unified_constraint", Tron1UnifiedRobot, Tron1ConstraintUnifiedCfg(), Tron1ConstraintUnifiedCfgPPO())
    args = get_args()
    play_on_constraint_policy_runner(args)