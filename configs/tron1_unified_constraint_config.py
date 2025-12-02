# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from global_config import ROOT_DIR
class Tron1ConstraintUnifiedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 117
        n_priv_latent =  4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
        # n_priv_latent = 3
        n_proprio = 33 + 4
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        delay_termination_time_s = 0.5
        
        dof_vel_use_pos_diff = True

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5] # [1.0, 2.5]
            offsets = [0.5, 0.5]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.10, 0.10] # [0.0, 0.1]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8 + 0.1664]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
                'joint_left_leg_1': 0,
                'joint_right_leg_1': 0,

                'joint_left_leg_2': 0.0,
                'joint_right_leg_2': 0.0,

                'joint_left_leg_3': -0.0,
                'joint_right_leg_3': -0.0,

                'joint_left_leg_4': 0,
                'joint_right_leg_4': 0,
        }

    class control( LeggedRobotCfg.control ):
        action_scale_pos = 0.5
        action_scale_vel = 0.5
        control_type = "P"
        stiffness = {
            "joint_left_leg_1": 42,
            "joint_right_leg_1": 42,
            "joint_left_leg_2": 42,
            "joint_right_leg_2": 42,
            "joint_left_leg_3": 42,
            "joint_right_leg_3": 42,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }  # [N*m/rad]
        damping = {
            "joint_left_leg_1": 2.5,
            "joint_right_leg_1": 2.5,
            "joint_left_leg_2": 2.5,
            "joint_right_leg_2": 2.5,
            "joint_left_leg_3": 2.5,
            "joint_right_leg_3": 2.5,
            "joint_left_leg_4": 1.5,
            "joint_right_leg_4": 1.5,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class sim( LeggedRobotCfg.sim ):
        dt = 0.0025
        # dt = 0.005

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 8  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        spin_high_speed = 10.0

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [0, 0]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14159, 3.14159]
            height = [0.65, 0.8]
            gait_mode = [0, 1]
            jump_flag = [0, 1]
            spin_flag = [0, 1]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/tron1/urdf/robot.urdf'
        foot_name = "leg_4"
        name = "tron1"
        foot_radius = 0.127
        penalize_contacts_on = ["leg_2","leg_3"]
        terminate_after_contacts_on = ["base","leg_1"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            #main rewards
            keep_balance = 1.0
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            torques = -0.00016
            dof_acc = -1.5e-7
            action_rate = -0.03
            action_smoothness = -0.03
            dof_pos_limits = -2.0
            collision = -50
            orientation = -20.0
            feet_distance = -20
            base_height_tracking = 2.0
            hip_symmetry = -1.0
            # tracking related rewards
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0

            # regulation related rewards
            #wheel
            leg_symmetry = -0.0
            same_foot_x_position = -0.0 # 0.5
            same_foot_z_position = -0.0
            tracking_contacts_shaped_force_wheel = -5.0
            tracking_contacts_shaped_vel_wheel = -5.0
            tracking_contacts_shaped_height_wheel = -5.0
            phase_wheel_vel_wheel = -2.0
            simple_coordination_wheel = -2.0
            no_fly_wheel= 2.0
            #biped
            tracking_lin_vel_feet = 4.0
            tracking_ang_vel_feet = 2.0
            no_fly = 2.0
            tracking_contacts_shaped_force = -5.0
            tracking_contacts_shaped_vel = -5.0
            tracking_contacts_shaped_height = -5.0
            phase_wheel_vel = -2.0
            simple_coordination = -2.0
            #jump
            fly = 2.0
            jump_landing = 2.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.6 + 0.1664
        feet_height_target = 0.10
        min_feet_distance = 0.32
        max_feet_distance = 0.35
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.0550
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

    class domain_rand( LeggedRobotCfg.domain_rand):
        radnomize_joint_friction = False
        radnomize_joint_damping = False
        randomize_friction = True
        friction_range = [0.2, 1.2]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 2]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.03, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        randomize_lag_timesteps = True
        # lag_timesteps = 4 #dt = 0.05
        lag_timesteps = 8 # dt = 0.025

        disturbance = False
        disturbancFe_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_deadband = True
        deadband_range = [0.8, 1.2]
        deadband = 0.05

    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class costs:
        class scales:
            hip_pos = 0.1
            same_foot_x_position = 0.1


        class d_values:
            hip_pos = 0.0
            same_foot_x_position = 0.0

    class cost:
        num_costs = 2

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    class symmetry:
        obs_permutation = {
            "policy": [
                0, 1, 2,          # base_ang_vel
                3, 4, 5,          # projected_gravity
                6, 7, 8,          # commands+
                13, 14, 15, 16, 9, 10, 11, 12,     # dof_pos
                21, 22, 23, 24, 17, 18, 19, 20,    # dof_vel
                29, 30, 31, 32, 25, 26, 27, 28     # action history
            ],  # 构造 33维 perm 索引

            "critic": [
                0, 1, 2,          # base_ang_vel
                3, 4, 5,          # projected_gravity
                6, 7, 8,          # commands
                13, 14, 15, 16, 9, 10, 11, 12,     # dof_pos
                21, 22, 23, 24, 17, 18, 19, 20,    # dof_vel
                29, 30, 31, 32, 25, 26, 27, 28     # action history
            ]   # 构造 546维 perm 索引
        }
        joint_permutation = [
            4, 5, 6, 7, 0, 1, 2, 3  # 轮腿左右交换动作顺序
        ]

class Tron1ConstraintUnifiedCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1
        # -- Symmetry Augmentation
        symmetry_cfg = {
            "use_data_augmentation": False,  # this adds symmetric trajectories to the batch
            "use_mirror_loss": False,       # this adds symmetry loss term to the loss function
            "data_augmentation_func": "utils.symmetry_utils:get_symmetric_states",
            "mirror_loss_coeff": 0.0,
        }

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 2

        teacher_act = True
        imi_flag = True
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'tron1_unified_barlowtwins'
        experiment_name = 'tron1_unified_constraint'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 30000
        num_steps_per_env = 24
        resume = True
        resume_path = 'logs/tron1_unified_constraint/Dec01_19-59-48_tron1_unified_barlowtwins/model_4400.pt'
