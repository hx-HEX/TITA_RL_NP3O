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
class TitaFusionCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 8192

        n_scan = 187
        n_priv_latent =  4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
        n_proprio = 33
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        delay_termination_time_s = 0.5

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x, y, z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x, y, z [rad/s]       
        default_joint_angles = {
                'joint_left_leg_1': 0,
                'joint_right_leg_1': 0,

                'joint_left_leg_2': 0.8,
                'joint_right_leg_2': 0.8,

                'joint_left_leg_3': -1.5,
                'joint_right_leg_3': -1.5,

                'joint_left_leg_4': 0,
                'joint_right_leg_4': 0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            "joint_left_leg_1": 40,
            "joint_right_leg_1": 40,
            "joint_left_leg_2": 40,
            "joint_right_leg_2": 40,
            "joint_left_leg_3": 40,
            "joint_right_leg_3": 40,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }  # [N*m/rad]
        damping = {
            "joint_left_leg_1": 2.0,
            "joint_right_leg_1": 2.0,
            "joint_left_leg_2": 2.0,
            "joint_right_leg_2": 2.0,
            "joint_left_leg_3": 2.0,
            "joint_right_leg_3": 2.0,
            "joint_left_leg_4": 1.5,
            "joint_right_leg_4": 1.5,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        action_scale_pos = 0.5
        action_scale_vel = 4.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5

        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-2, 2]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):

        file = '{ROOT_DIR}/resources/tita/urdf/tita_description.urdf'
        foot_name = "leg_4"
        name = "tita"
        penalize_contacts_on = ["leg_3"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        min_feet_distance = 0.42
        max_feet_distance = 0.46
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = -0.0
            termination = -200
            tracking_lin_vel = 4.0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.05
            dof_vel = 0.0
            dof_acc = -1.5e-7
            base_height = -5.0
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = 0.0
            action_rate = -0.01
            action_smoothness= -0.01
            stand_still = -1.0
            foot_clearance= -0.0
            orientation=-10.0
            feet_distance = -5.0

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.0, 1.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 5]
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
        lag_timesteps = 3

        disturbance = True
        disturbancFe_range = [-30.0, 30.0]
        disturbance_interval = 8
    
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
            # torques = 0.3
            # powers = 0.3
            # pos_limit = 0.1
            # torque_limit = 0.3
            # # dof_vel_limits = 0.3
            # acc_smoothness = 0.1
            # # collision = 0.1
            # feet_contact_forces = 0.1
            # stumble = 0.0001
            hip_pos = 0.01

        class d_values:
            # torques = 0.0
            # powers = 0.0
            # pos_limit = 0.0
            # torque_limit = 0.0
            # # dof_vel_limits = 0.0
            # acc_smoothness = 0.0
            # # collision = 0.0
            # feet_contact_forces = 0.0
            # stumble = 0.0
            hip_pos = 0.0

    class cost:
        num_costs = 1
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False
        terrain_proportions = [0.1, 0.2, 0.4, 0.2, 0.0, 0.0, 0.1]

    class symmetry:
        obs_permutation = {
            "policy": [
                0, 1, 2,          # base_ang_vel
                3, 4, 5,          # projected_gravity
                6, 7, 8,          # commands
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

class TitaFusionCfgPPO( LeggedRobotCfgPPO ):
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
            "use_data_augmentation": True,  # this adds symmetric trajectories to the batch
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
        num_costs = 1

        teacher_act = True
        imi_flag = True
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'test_barlowtwins_fusion'
        experiment_name = 'tita_fusion_constraint'
        policy_class_name = 'FusionPolicyWithCritic' 
        runner_class_name = 'OnConstraintPolicyRunnerFusion'
        algorithm_class_name = 'NP3OFusion'
        max_iterations = 10000
        num_steps_per_env = 24
        resume = False
        resume_path = 'logs/tita_fusion_constraint/Aug24_23-10-10_test_barlowtwins_fusion/model_10000.pt'
        experts = {
            "wheel": {
                "cfg_path": "logs/tita_wheel_constraint/Sep03_01-07-07_test_barlowtwins_feetcontact/tita_wheel_constraint_config.py",
                "ckpt_path": "logs/tita_wheel_constraint/Sep03_01-07-07_test_barlowtwins_feetcontact/model_10000.pt"
            },
            "biped": {
                    "cfg_path": "logs/tita_feet_constraint/Sep02_19-50-50_test_barlowtwins_feetcontact/tita_feet_constraint_config.py",
                    "ckpt_path": "logs/tita_feet_constraint/Sep02_19-50-50_test_barlowtwins_feetcontact/model_4800.pt"
            }
        }