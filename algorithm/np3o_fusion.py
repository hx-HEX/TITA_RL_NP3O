import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules.actor_critic import ActorCriticRMA
from runner.rollout_storage_fusion import RolloutStorageWithCostFusion
from utils import unpad_trajectories,string_to_callable
from envs.vec_env import VecEnv

class NP3OFusion:
    actor_critic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 k_value,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 cost_value_loss_coef=1.0,
                 cost_viol_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 # Symmetry parameters
                 symmetry_cfg: dict = None,
                 **kwargs
                 ):

        
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.print_cnt = 0
        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        if hasattr(self.actor_critic, 'imitation_learning_loss') and self.actor_critic.imi_flag:
            self.imi_flag = True
            print('running with imi loss on')
        else:
            self.imi_flag = False
            print('running with imi loss off')

        self.imi_weight = 1

        # self.imitation_params_list = list(self.actor_critic.actor_student_backbone.parameters())
        # self.imitation_optimizer = optim.Adam(self.imitation_params_list, lr=3e-4)
        self.transition = RolloutStorageWithCostFusion.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.k_value = k_value

        self.substeps = 1

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,cost_shape,cost_d_values):
        self.storage = RolloutStorageWithCostFusion(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape,cost_shape,cost_d_values,self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def set_imi_flag(self,flag):
        self.imi_flag = flag
        if self.imi_flag:
            print("runing with imitation")
        else:
            print("runing without imitation")
    
    def set_imi_weight(self,value):
        self.imi_weight = value

    def act(self, obs, critic_obs, info):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, rewards_wheel, rewards_biped, costs, dones, dones_wheel, dones_biped, infos):

        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        self.transition.rewards_wheel = rewards_wheel.clone()
        self.transition.rewards_biped = rewards_biped.clone()
        self.transition.dones_wheel = dones_wheel.clone()
        self.transition.dones_biped = dones_biped.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
        if 'terrain_levels' in infos:
            self.transition.terrain_levels = infos['terrain_levels'].unsqueeze(1).to(self.device)
        if 'terrain_types' in infos:
            self.transition.terrain_types = infos['terrain_types'].unsqueeze(1).to(self.device)
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, obs):
        last_cost_values = self.actor_critic.evaluate_cost(obs).detach()
        self.storage.compute_cost_returns(last_cost_values,self.gamma,self.lam)

    def compute_surrogate_loss(self,actions_log_prob_batch,old_actions_log_prob_batch,advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss
    
    def compute_cost_surrogate_loss(self,actions_log_prob_batch,old_actions_log_prob_batch,cost_advantages_batch):
        # cost_advantages_batch : batch_size,num_type_costs
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        # (batch_size,num_type_costs) * (batch_size,1) = (batch_size,num_type_costs)
        surrogate = cost_advantages_batch*ratio.view(-1,1)
        surrogate_clipped = cost_advantages_batch*torch.clamp(ratio.view(-1,1), 1.0 - self.clip_param,1.0 + self.clip_param)
        # num_type_costs
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(0)
        return surrogate_loss
    
    def compute_value_loss(self,target_values_batch,value_batch,returns_batch):
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss
    
    def update_k_value(self,i):
        self.k_value = torch.min(torch.ones_like(self.k_value),self.k_value*(1.0004**i))
        return self.k_value
    
    def compute_viol(self,actions_log_prob_batch,old_actions_log_prob_batch,cost_advantages_batch,cost_volation_batch):

        # compute cliped cost advantage
        cost_surrogate_loss = self.compute_cost_surrogate_loss(actions_log_prob_batch=actions_log_prob_batch,
                                                          old_actions_log_prob_batch=old_actions_log_prob_batch,
                                                          cost_advantages_batch=cost_advantages_batch)
        # compute the violation term,d_values :(num_type_costs)
        # cost_volation = (1-self.gamma)*(torch.squeeze(cost_returns_batch).mean() - self.d_values)
        cost_volation_loss = cost_volation_batch.mean()
        # combine the result
        cost_loss = cost_surrogate_loss + cost_volation_loss
        # do max and sum over
        #cost_loss = self.k_value*torch.sum(F.relu(cost_loss))
        cost_loss = torch.sum(self.k_value*F.relu(cost_loss))
        return cost_loss

    def update(self):
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_viol_loss = 0
        mean_surrogate_loss = 0
        mean_imitation_loss = 0
        mean_pri_loss = 0
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # ---------------- NEW: 统计门控损失均值（不改变函数返回签名，不强制用） ----------------
        mean_gating_reg_loss = 0.0
        mean_gating_sup_loss = 0.0
        # -------------------------------------------------------------------------

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_cost_values_batch,cost_advantages_batch,cost_returns_batch,cost_violation_batch,terrain_levels_batch,terrain_types_batch in generator:

                # number of augmentations per sample
                # we start with 1 and increase it if we use symmetry augmentation
                num_aug = 1
                # original batch size
                original_batch_size = obs_batch.shape[0]
                # Perform symmetric augmentation
                if self.symmetry and self.symmetry["use_data_augmentation"]:
                    # augmentation using symmetry
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    # returned shape: [batch_size * num_aug, ...]
                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                    )
                    critic_obs_batch, _ = data_augmentation_func(
                        obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    # repeat the rest of the batch,reward
                    # -- actor
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    # -- critic
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)
                    # repeat cost-related targets
                    target_cost_values_batch = target_cost_values_batch.repeat(num_aug, 1)
                    cost_advantages_batch = cost_advantages_batch.repeat(num_aug, 1)
                    cost_returns_batch = cost_returns_batch.repeat(num_aug, 1)
                    cost_violation_batch = cost_violation_batch.repeat(num_aug, 1)  # 如果有这个量并用于loss

                # -- actor
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # -- critic
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                # -- entropy
                # we only keep the entropy of the first augmentation (the original one)
                mu_batch = self.actor_critic.action_mean[:original_batch_size]
                sigma_batch = self.actor_critic.action_std[:original_batch_size]
                entropy_batch = self.actor_critic.entropy[:original_batch_size]
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                surrogate_loss = self.compute_surrogate_loss(actions_log_prob_batch=actions_log_prob_batch,
                                                         old_actions_log_prob_batch=old_actions_log_prob_batch,
                                                         advantages_batch=advantages_batch)

                # Cost voilation
                viol_loss = self.compute_viol(actions_log_prob_batch=actions_log_prob_batch,
                                old_actions_log_prob_batch=old_actions_log_prob_batch,
                                cost_advantages_batch=cost_advantages_batch,
                                cost_volation_batch=cost_violation_batch)
                # value function loss
                value_loss = self.compute_value_loss(target_values_batch=target_values_batch,
                                        value_batch=value_batch,
                                        returns_batch=returns_batch)
                
                # Cost value function loss
                cost_value_loss = self.compute_value_loss(target_values_batch=target_cost_values_batch,
                                                        value_batch=cost_value_batch,
                                                        returns_batch=cost_returns_batch)

                main_loss = surrogate_loss + self.cost_viol_loss_coef * viol_loss 
                combine_value_loss = self.cost_value_loss_coef * cost_value_loss + self.value_loss_coef * value_loss
                entropy_loss = - self.entropy_coef * entropy_batch.mean()

                # ---------------------- NEW: 门控损失（正则 + 可选监督） ----------------------
                gating_reg = 0.0
                gating_sup = 0.0
                weights = getattr(self.actor_critic, "last_attention_weights", None)
                cluster_probs_debug = getattr(self.actor_critic, "cluster_probs_debug", None)
                # ---- b) 基于 terrain-level + advantage 的监督项 ----
                if getattr(self.actor_critic, "gating_sup_coef", 0.0) > 0.0:
                    B, N = weights.shape
                    device = obs_batch.device

                    # terrain level 和 type
                    terrain_level = terrain_levels_batch.to(device=device, dtype=torch.float32)
                    terrain_types = terrain_types_batch.to(device=device, dtype=torch.float32)

                    # 条件 mask
                    mask_wheel = (
                        (terrain_types < 6)
                        | (((terrain_types >= 6) & (terrain_types < 14)) & (terrain_level < 4))
                        | ((terrain_types >= 14) & (terrain_types < 18)) 
                        # | ((terrain_types >= 18) & (terrain_level < 5))
                        | ((terrain_types >= 18))
                    )
                    mask_wheel = mask_wheel.squeeze(-1)  # 变成 (B,)
                    mask_biped = ~mask_wheel

                    # 初始化 soft target
                    terrain_soft = torch.zeros((B, 2), device=device, dtype=torch.float32)

                    terrain_soft[mask_wheel, :] = torch.tensor([1.0, 0.0], device=device)
                    terrain_soft[mask_biped, :] = torch.tensor([0.0, 1.0], device=device)

                    # 如果有数据增强（num_aug > 1），扩展一下
                    terrain_soft = terrain_soft.repeat(num_aug, 1)  # (B*num_aug, 2)

                    # 3) advantage-based soft target
                    with torch.no_grad():
                        adv_wheel = self.actor_critic.expert_wheel.evaluate(
                            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                        ).squeeze(-1)
                        adv_biped = self.actor_critic.expert_biped.evaluate(
                            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                        ).squeeze(-1)
                        adv_experts = torch.stack([adv_wheel, adv_biped], dim=1)
                        adv_experts = adv_experts - adv_experts.mean(dim=1, keepdim=True)
                        adv_soft = F.softmax(adv_experts, dim=1)  # shape (B, 2)

                    # 4) 融合 terrain 与 advantage
                    alpha = 0.0
                    w_target = alpha * adv_soft + (1 - alpha) * terrain_soft  # (B, 2)
                    device = weights.device
                    # 5) KL 散度损失
                    weights_prob = (weights + 1e-8) / (weights + 1e-8).sum(dim=1, keepdim=True)
                    weights_log = weights_prob.log()
                    w_target = (w_target + 1e-8) / (w_target + 1e-8).sum(dim=1, keepdim=True)
                    w_target = w_target.to(device)
                    gating_sup = F.kl_div(weights_log, w_target, reduction='batchmean')
                    if(self.print_cnt > 23):
                        # ✅ 打印关键信息
                        print("[Gating Debug] weights (first 5):", weights[:5])
                        print("[Gating Debug] w_target (first 5):", w_target[:5])
                        print("[Gating Debug] gating_sup loss:", gating_sup.item())
                        import numpy as np
                        np.set_printoptions(precision=3, suppress=True)
                        print("[Gating Debug] cluster_probs_debug:", cluster_probs_debug[:5].cpu().numpy())

                        self.print_cnt = 0
                    else:
                        self.print_cnt = self.print_cnt + 1
                    
                else:
                    gating_sup = torch.tensor(0.0, device=obs_batch.device)
                
                if self.imi_flag:
                    imitation_loss, pri_loss = self.actor_critic.imitation_learning_loss(obs_batch)
                    loss = main_loss + combine_value_loss + entropy_loss \
                        + self.imi_weight * imitation_loss \
                        + gating_reg + gating_sup
                    # loss = self.imi_weight * imitation_loss \
                    #     + gating_reg + gating_sup
                else:
                    loss = main_loss + combine_value_loss + entropy_loss \
                        + gating_reg + gating_sup
                    # loss = gating_reg + gating_sup

                # Symmetry loss
                if self.symmetry:
                    # obtain the symmetric actions
                    # if we did augmentation before then we don't need to augment again
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        obs_batch, _ = data_augmentation_func(
                            obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                        )
                        # compute number of augmentations per sample
                        num_aug = int(obs_batch.shape[0] / original_batch_size)

                    # actions predicted by the actor for symmetrically-augmented observations
                    mean_actions_batch = self.actor_critic.act_teacher(obs_batch.detach().clone())

                    # compute the symmetrically augmented actions
                    # note: we are assuming the first augmentation is the original one.
                    #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                    #   However, the symmetry loss is computed using the mean of the distribution.
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                    )

                    # compute the loss (we skip the first augmentation as it is the original one)
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    # add the loss to the total loss
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_viol_loss += viol_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if self.imi_flag:
                    mean_imitation_loss += imitation_loss.item()
                    mean_pri_loss += pri_loss.item()
                else:
                    mean_imitation_loss += 0
                    mean_pri_loss +=0
                # -- Symmetry loss
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()

                mean_gating_reg_loss   += float(gating_reg) if torch.is_tensor(gating_reg) else gating_reg
                mean_gating_sup_loss   += float(gating_sup) if torch.is_tensor(gating_sup) else gating_sup

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_viol_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates*self.substeps
        mean_pri_loss /= num_updates*self.substeps
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        mean_gating_reg_loss   /= num_updates
        mean_gating_sup_loss   /= num_updates

        self.storage.clear()

        return mean_value_loss,mean_cost_value_loss,mean_viol_loss,mean_surrogate_loss,mean_imitation_loss,mean_symmetry_loss,mean_pri_loss,mean_gating_reg_loss,mean_gating_sup_loss
    
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            self.depth_actor_optimizer.zero_grad()
            depth_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_actor_loss.item()