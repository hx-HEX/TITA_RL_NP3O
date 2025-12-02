import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F 
from modules.common_modules import AutoEncoder, BetaVAE, RnnBarlowTwinsStateHistoryEncoder, RnnStateHistoryEncoder, StateHistoryEncoder, get_activation, mlp_factory, mlp_layernorm_factory
from modules.transformer_modules import StateCausalTransformer
class Config:
    def __init__(self):
        self.n_obs = 45
        self.block_size = 9
        self.n_action = 45+3
        self.n_layer: int = 4
        self.n_head: int = 4
        self.n_embd: int = 32
        self.dropout: float = 0.0
        self.bias: bool = True

class CnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 priv_encoder_output_dim,
                 actor_hidden_dims=[256, 256, 256],
                 activation='elu'):
        super(CnnActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.priv_encoder_output_dim = priv_encoder_output_dim
        self.activation = activation
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        self.actor_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        self.actor = nn.Sequential(*self.actor_layers)
    
    def forward(self,obs,hist):
        latent = self.history_encoder(hist)
        backbone_input = torch.cat([obs,latent], dim=1)
        mean = self.actor(backbone_input)
        return mean
    
class RnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 encoder_dims,
                 decoder_dims,
                 actor_dims,
                 encoder_output_dim,
                 hidden_dim,
                 num_actions,
                 activation,) -> None:
        super(RnnActor,self).__init__()
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=activation,
                                                  input_size=num_prop,
                                                  encoder_dims=encoder_dims,
                                                  hidden_size=hidden_dim,
                                                  output_size=encoder_output_dim)
        self.next_state_decoder =nn.Sequential(*mlp_factory(activation=activation,
                                              input_dims=hidden_dim,
                                              out_dims=num_prop+7,
                                              hidden_dims=decoder_dims))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=hidden_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        latents = self.rnn_encoder(obs_hist_full)
        actor_input = torch.cat([latents[:,-1,:],obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    def predict_next_state(self,obs_hist):
        # self.rnn_encoder.reset_hidden()
        latents = self.rnn_encoder(obs_hist)
        predicted = self.next_state_decoder(latents[:,-1,:])
        return predicted
    
class RnnBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 obs_encoder_dims,
                 rnn_encoder_dims,
                 actor_dims,
                 encoder_output_dim,
                 hidden_dim,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(RnnBarlowTwinsActor,self).__init__()
        self.rnn_encoder = RnnBarlowTwinsStateHistoryEncoder(activation_fn=activation,
                                                  input_size=num_prop,
                                                  hidden_size=hidden_dim,
                                                  output_size=encoder_output_dim,
                                                  final_output_size=latent_dim,
                                                  encoder_dims=rnn_encoder_dims)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        latents = self.rnn_encoder(obs_hist_full)
        actor_input = torch.cat([latents,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,weight):
        b = obs.size()[0]
        hist_latent = self.rnn_encoder(obs_hist)
        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)
        # c = torch.sum(c,dim=0,keepdim=True)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + weight*off_diag
        return loss
    
class MlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBarlowTwinsActor,self).__init__()
        self.mlp_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                #  out_dims=latent_dim+3,
                                out_dims=latent_dim+10,
                                 hidden_dims=mlp_encoder_dims))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                #  input_dims=latent_dim + num_prop + 3,
                                 input_dims=latent_dim + num_prop + 10,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        obs_hist_full = obs_hist_full[:,0:,:].view(b,-1)
        latents = self.mlp_encoder(obs_hist_full)
        actor_input = torch.cat([latents,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        obs_hist = obs_hist[:,0:,:].view(b,-1)
        predicted = self.mlp_encoder(obs_hist)
        # hist_latent = predicted[:,3:]
        # priv_latent = predicted[:,:3]
        hist_latent = predicted[:,10:]
        priv_latent = predicted[:,:10]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss,priv_loss

class TransBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 obs_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(TransBarlowTwinsActor,self).__init__()
        self.transformer_config = Config()
        self.transformer_config.n_action=latent_dim + 7
        self.transformer_config.n_layer = 2
        
        self.trans_encoder = StateCausalTransformer(self.transformer_config)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 7,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
        ], dim=1)
        latent = self.trans_encoder(obs_hist_full[:,5:,:])
        actor_input = torch.cat([latent,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        predicted = self.trans_encoder(obs_hist[:,5:,:])
        hist_latent = predicted[:,7:]
        priv_latent = predicted[:,:7]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

class AeActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 encoder_dims,
                 decoder_dims,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(AeActor,self).__init__()
        self.ae = AutoEncoder(activation_fn=activation,
                            input_size=num_prop*num_hist,
                            encoder_dims=encoder_dims,
                            decoder_dims=decoder_dims,
                            latent_dim=latent_dim,
                            output_size=num_prop)
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

    def forward(self,obs,obs_hist):
        # self.rnn_encoder.reset_hidden()
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        b,t,n = obs_hist_full.size()
        obs_hist_full = obs_hist_full.view(b,-1)
        latent = self.ae.encode(obs_hist_full)
        actor_input = torch.cat([latent,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    def predict_next_state(self,obs_hist):
        b,t,n = obs_hist.size()
        obs_hist_flatten = obs_hist.view(b,-1)
        latent = self.ae.encode(obs_hist_flatten)
        predicted = self.ae.decode(latent)
        return predicted,latent
        
class StateCausalTransformerActor(nn.Module):
    def __init__(self):
        super(StateCausalTransformerActor,self).__init__()
        self.transformer_config = Config()
        self.transformer = StateCausalTransformer(self.transformer_config)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        
        predicted_state = self.transformer(obs_hist_full[:,5:,:])
        action = predicted_state[:,36:]
        return action
    
    def predict_next_state(self,obs_hist):
        predicted_state = self.transformer(obs_hist[:,5:,:])
        return predicted_state

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with student actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)
        # actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+32,num_actions,actor_hidden_dims,last_act=False)

        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)
        self.actor_student_backbone = CnnActor(num_prop=num_prop,
                                               num_hist=num_hist,
                                               num_actions=num_actions,
                                               priv_encoder_output_dim=priv_encoder_output_dim,
                                               actor_hidden_dims=actor_hidden_dims,
                                               activation=activation)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        if self.teacher_act:
            mean = self.act_teacher(obs)
        else:
            mean = self.act_student(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        hist = obs[:, -self.num_hist*self.num_prop:].view(-1,self.num_hist,self.num_prop)
        mean = self.actor_student_backbone(obs_prop,hist)
        return mean
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]

        # scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,hist_latent], dim=1)
        mean = self.actor_teacher_backbone(backbone_input)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
     
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        with torch.no_grad():
            target_mean = self.act_teacher(obs)
        mean = self.act_student(obs)

        loss = F.mse_loss(mean,target_mean.detach())
        return loss
    
    def imitation_mode(self):
        self.actor_teacher_backbone.eval()
        self.scan_encoder.eval()
        self.priv_encoder.eval()
    
    def save_torch_jit_policy(self,path,device):
        print("ActorCriticRMA")
        obs_demo_input = torch.randn(1,self.num_prop).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
        model_jit = torch.jit.trace(self.actor_student_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

class ActorCriticBarlowTwins(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticBarlowTwins, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        # self.actor_teacher_backbone = RnnBarlowTwinsActor(num_prop=num_prop,
        #                               num_actions=num_actions,
        #                               actor_dims=[512,256,128],
        #                               encoder_output_dim=32,
        #                               hidden_dim=128,
        #                               activation=activation,
        #                               latent_dim=64,
        #                               obs_encoder_dims=[256,128],
        #                               rnn_encoder_dims=[128])
        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpBarlowTwinsActor(num_prop=num_prop,
                                      num_hist=10,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[256,128])
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        # priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 3]#只估线速度
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 10]#只估线速度
        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        return loss
    
    def imitation_mode(self):
        pass
    
    # def save_torch_jit_policy(self,path,device):
    #     obs_demo_input = torch.randn(1,self.num_prop).to(device)
    #     hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
    #     model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
    #     model_jit.save(path)
    #     torch_out = torch.onnx.export(self.actor_teacher_backbone,
    #                         (obs_demo_input,hist_demo_input),
    #                         "test.onnx",
    #                         verbose=True,
    #                         export_params=True
    #                         )
    def save_torch_jit_policy(self, path, device):
        obs_demo_input = torch.randn(1, self.num_prop, device=device)
        hist_demo_input = torch.randn(1, self.num_hist, self.num_prop, device=device)

        # 保存 TorchScript
        model_jit = torch.jit.trace(self.actor_teacher_backbone, (obs_demo_input, hist_demo_input))
        model_jit.save(path)
        print(f"[✓] TorchScript policy saved to {path}")

        # 导出 ONNX，固定输入输出名字
        torch.onnx.export(
            self.actor_teacher_backbone,
            (obs_demo_input, hist_demo_input),
            "test.onnx",
            verbose=True,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["obs", "obs_hist"],   # 👈 改为固定名字
            output_names=["action"],           # 👈 改为固定名字
            dynamic_axes=None                  # 固定 batch=1
        )
        print(f"[✓] ONNX policy saved to test.onnx")

class MlpBarlowTwinsEncoder(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 latent_dim,
                 activation,
                 num_prototypes=16,
                 normalize_latent=True) -> None:
        super(MlpBarlowTwinsEncoder, self).__init__()
        
        # 编码历史序列 obs_hist → latent
        self.mlp_encoder = nn.Sequential(
            *mlp_layernorm_factory(
                activation=activation,
                input_dims=num_prop * num_hist,
                out_dims=latent_dim,
                hidden_dims=mlp_encoder_dims
            )
        )
        
        # 编码当前帧 obs → latent
        self.obs_encoder = nn.Sequential(
            *mlp_layernorm_factory(
                activation=activation,
                input_dims=num_prop,
                out_dims=latent_dim,
                hidden_dims=obs_encoder_dims
            )
        )
        
        # 用于 Barlow Twins 对比学习的批归一化
        self.bn = nn.BatchNorm1d(latent_dim, affine=False)

        # 自监督聚类用 prototype
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))  # (K, D)
        self.normalize_latent = normalize_latent
        self.temperature = 0.1  # 可调超参

    def reset(self, dones=None):
        pass

    def forward(self, obs_hist_full):
        """
        输入:obs_hist_full: (B, num_prop * num_hist)
        输出:latent 向量: (B, D)
        """
        return self.mlp_encoder(obs_hist_full)

    def get_latent(self, obs_hist):
        """
        输入:obs_hist: (B, H, D_prop)
        输出:latent: (B, D)
        """
        B = obs_hist.size(0)
        obs_hist_flat = obs_hist.view(B, -1)
        return self.forward(obs_hist_flat)

    def get_cluster_probs(self, latent):
        """
        输入 latent: (B, D)
        输出 cluster_probs: (B, K)
        表示该样本在 K 个 prototype 下的 soft assignment。
        """
        if self.normalize_latent:
            latent = F.normalize(latent, dim=-1)            # (B, D)
            proto = F.normalize(self.prototypes, dim=-1)    # (K, D)
        else:
            proto = self.prototypes                         # (K, D)

        logits = latent @ proto.T / self.temperature        # (B, K)
        cluster_probs = F.softmax(logits, dim=-1)
        return cluster_probs

    def forward_cluster_probs(self, obs_hist):
        """
        输入 obs_hist: (B, H, D_prop)
        输出 cluster_probs: (B, K)
        """
        latent = self.get_latent(obs_hist)
        return self.get_cluster_probs(latent)

    def BarlowTwinsLoss(self, obs, obs_hist, weight):
        """
        obs: (B, D_prop)
        obs_hist: (B, H, D_prop)
        """
        B = obs.size(0)
        obs_hist_flat = obs_hist.view(B, -1)

        hist_latent = self.mlp_encoder(obs_hist_flat)  # (B, D)
        obs_latent = self.obs_encoder(obs)             # (B, D)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(B)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow(2).sum()

        loss = on_diag + weight * off_diag
        return loss

    
class HybridAttention(nn.Module):
    def __init__(self, latent_dim, num_experts=2, hidden_dim=None, temperature=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim or latent_dim

        self.query_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, self.hidden_dim)
        )

        # ✅ 使用可学习的 expert key 向量（不从输入中提取）
        self.key_embeds = nn.Parameter(torch.randn(num_experts, latent_dim))
        self.key_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, self.hidden_dim)
        )

        # 温度系数（可用于 softmax 的缩放）
        self.register_buffer("temperature", torch.tensor(float(temperature) if temperature is not None else latent_dim ** 0.5))

        # 👉 如果需要调试输出 attention，可以临时存储 debug 信息
        self.last_attention_weights_debug = None

    def forward(self, query_latent, eps=0.0, min_w=0.0):
        """
        Args:
            query_latent: (B, D)
            eps: epsilon-greedy mixing  (0~0.2)
            min_w: 每个 expert 最小权重下界
        Returns:
            weights: (B, N)
        """
        B, D = query_latent.shape
        N = self.num_experts

        q = self.query_proj(query_latent).unsqueeze(1)  # (B,1,H)

        # ✅ 用 learnable key 向量扩展为 (B, N, D)
        expert_keys = self.key_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        k = self.key_proj(expert_keys)  # (B, N, H)

        # Dot-product attention
        scores = torch.sum(q * k, dim=-1) / self.temperature  # (B,N)
        weights = F.softmax(scores, dim=-1)                   # (B,N)

        # 1) epsilon-greedy 探索（平滑到均匀分布）
        if eps > 0.0:
            weights = (1.0 - eps) * weights + eps * (1.0 / N)

        # 2) 托底约束：保证每个 expert 最小权重
        if min_w > 0.0:
            weights = torch.clamp(weights, min=min_w)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # ✅ 保留梯度的 attention weights（用于后续监督 loss）
        self.last_attention_weights = weights  # 不做 .detach()

        # （可选）debug 打印用的无梯度版本
        self.last_attention_weights_debug = weights.detach().cpu()

        return weights

class GatingNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, num_experts=2, temperature=None,num_prototypes=16):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature or 1.0  # 可用于退火
        self.net = nn.Sequential(
            nn.Linear(num_prototypes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.last_attention_weights_debug = None  # for debug

    def forward(self, latent, eps=0.0, min_w=0.0):
        """
        Args:
            latent: 本体感知编码 (B, D)
            eps: epsilon-greedy
            min_w: expert 最小权重
        Returns:
            weights: (B, N)
        """
        logits = self.net(latent) / self.temperature
        weights = F.softmax(logits, dim=-1)

        if eps > 0.0:
            weights = (1.0 - eps) * weights + eps * (1.0 / self.num_experts)

        if min_w > 0.0:
            weights = torch.clamp(weights, min=min_w)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # ✅ 保留梯度
        self.last_attention_weights = weights
        self.last_attention_weights_debug = weights.detach().cpu()

        return weights

class FusionPolicyWithCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  expert_wheel, 
                        expert_biped, 
                        num_prop, 
                        num_scan,
                        num_priv_latent, 
                        num_hist,
                        latent_dim, 
                        action_dim, 
                        scan_encoder_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                         # ------ 新增的一些超参(可用默认值) ------
                        gating_lb_coef=1e-2,          # load-balance 系数
                        gating_ent_coef=5e-4,         # 熵正则系数
                        gating_sup_coef=1.0,          # (可选) 监督 loss 系数
                        gating_tau=1.0,               # (可选) 监督 softmax 温度
                        gating_eps_init=0.2,          # 前期 epsilon-greedy
                        gating_eps_final=0.0,
                        gating_min_w=0.00,            # 每个 expert 最小权重
                        temperature_init=None,        # attention 温度(可退火)
                        **kwargs):
        
        super().__init__()
        self.kwargs = kwargs
        self.gating_lb_coef = gating_lb_coef
        self.gating_ent_coef = gating_ent_coef
        self.gating_sup_coef = gating_sup_coef
        self.gating_tau = gating_tau
        self.gating_eps = gating_eps_init
        self.gating_eps_init = gating_eps_init
        self.gating_eps_final = gating_eps_final
        self.gating_min_w = gating_min_w
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_priv_latent = num_priv_latent
        self.expert_wheel = expert_wheel
        self.expert_biped = expert_biped
        # 训练中可在 runner 里调用 set_temperature / set_eps 做退火
        # self.attention = HybridAttention(latent_dim, hidden_dim=64,
        #                                  temperature=temperature_init)
        self.gating_net = GatingNetwork(latent_dim, hidden_dim=64, temperature=temperature_init)

        self.std = nn.Parameter(torch.ones(action_dim))
        self.distribution = None
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        activation = get_activation(activation)

        for param in self.expert_wheel.parameters():
            param.requires_grad = False
        for param in self.expert_biped.parameters():
            param.requires_grad = False

        self.main_encoder = MlpBarlowTwinsEncoder(num_prop=num_prop,
                                      num_hist=10,
                                      obs_encoder_dims=[256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      latent_dim=16,
                                      activation=activation
                                      )
        
        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

    def reset(self, dones=None):
        pass

    def forward(self, obs, obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        obs_hist_full = obs_hist_full[:,0:,:].view(b,-1)

        with torch.no_grad():
            latent_wheel = self.expert_wheel.actor_teacher_backbone.mlp_encoder(obs_hist_full)[..., 3:]
            latent_biped = self.expert_biped.actor_teacher_backbone.mlp_encoder(obs_hist_full)[..., 3:]
        
        # latents = torch.stack([latent_wheel, latent_biped], dim=1)  # (B, 2, D)
        # weights = self.attention(
        #     latent_main,
        #     eps=self.gating_eps,
        #     min_w=self.gating_min_w
        # )
        latent_main = self.main_encoder.forward_cluster_probs(obs_hist_full)
        weights = self.gating_net(
            latent_main,
            eps=self.gating_eps,
            min_w=self.gating_min_w
        )
        self.last_attention_weights = weights
        self.last_attention_weights_debug = weights.detach().cpu()
        self.cluster_probs_debug = latent_main.detach().cpu()

        with torch.no_grad():
            action_wheel = self.expert_wheel.actor_teacher_backbone(obs, obs_hist)
            action_biped = self.expert_biped.actor_teacher_backbone(obs, obs_hist)

        actions = torch.stack([action_wheel, action_biped], dim=1)  # (B, 2, A)
        fusion_action = torch.sum(actions * weights.unsqueeze(-1), dim=1)  # (B, A)
        return fusion_action, weights

    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean, weight = self.forward(obs_prop, obs_hist)
        return mean
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)

        loss = self.main_encoder.BarlowTwinsLoss(obs_prop,obs_hist,5e-3)
        return loss,loss
    
    #critic
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.critic(backbone_input)
        return value

    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_mode(self):
        pass

    def save_torch_jit_policy(self, path_prefix, device):
        """保存专门给 TensorRT 用的 TorchScript 和 ONNX(固定 batch=1)"""
        class _ExportWrapper(nn.Module):
            def __init__(self, fusion_policy):
                super().__init__()
                self.fusion_policy = fusion_policy

            def forward(self, obs, obs_hist):
                # 只返回动作，不要权重
                action, _ = self.fusion_policy.forward(obs, obs_hist)
                return action.to(torch.float32)

        export_module = _ExportWrapper(self).to(device)
        export_module.eval()

        # 固定 batch=1 的输入（不要 dynamic_axes）
        obs_demo_input = torch.randn(1, self.num_prop, device=device)
        hist_demo_input = torch.randn(1, self.num_hist, self.num_prop, device=device)

        # TorchScript 保存
        traced = torch.jit.trace(export_module, (obs_demo_input, hist_demo_input))
        traced.save(f"{path_prefix}_trt.pt")
        print(f"[✓] TorchScript policy (TRT) saved to {path_prefix}_trt.pt")

        # ONNX 导出（固定 shape，不用 dynamic_axes）
        torch.onnx.export(
            export_module,
            (obs_demo_input, hist_demo_input),
            f"{path_prefix}_trt.onnx",
            input_names=["obs", "obs_hist"],
            output_names=["action"],
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamic_axes=None  # 固定 batch=1
        )
        print(f"[✓] ONNX policy (TRT) saved to {path_prefix}_trt.onnx")

# ----------------- 正则：均衡 + 熵 -----------------
    def gating_reg_loss(self, weights):
        """
        weights: (B, N)
        """
        if self.gating_lb_coef <= 0 and self.gating_ent_coef <= 0:
            return weights.new_zeros(())
        # load-balance：让 batch 内的平均权重靠近均匀
        mean_w = weights.mean(dim=0)                            # (N,)
        target = torch.full_like(mean_w, 1.0 / mean_w.numel())
        lb_loss = torch.sum((mean_w - target) ** 2)

        # 熵正则：防止太尖锐
        ent = - (weights * (weights.clamp_min(1e-8).log())).sum(dim=-1).mean()
        ent_loss = -ent  # 想让熵大 => 惩罚负熵

        return self.gating_lb_coef * lb_loss + self.gating_ent_coef * ent_loss

    # ----------------- (可选)门控监督：用 advantage / values 引导 -----------------
    def gating_supervision_loss(self, weights, expert_values):
        """
        weights: (B, N)
        expert_values: (B, N)  # 每个 expert 对当前obs的“好坏”度量,通常由 runner 算好传进来（比如 advantage)
        """
        if self.gating_sup_coef <= 0.0:
            return weights.new_zeros(())
        with torch.no_grad():
            target = torch.softmax(expert_values / self.gating_tau, dim=-1)  # (B,N)
        # KL(target || weights)
        loss = torch.sum(target * (target.clamp_min(1e-8).log() - weights.clamp_min(1e-8).log()), dim=-1).mean()
        return self.gating_sup_coef * loss

    # 供 runner 做简单退火/调度
    # def set_temperature(self, new_temp: float):
    #     self.attention.temperature.fill_(new_temp)

    def set_temperature(self, new_temp: float):
        self.gating_net.temperature = new_temp

    def set_eps(self, new_eps: float):
        self.gating_eps = new_eps

class ActorCriticUnified(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        cmd_dim,  
                        num_primitives=5,
                        primitive_dim=16,
                        obs_latent_dim=16,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticUnified, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.cmd_dim = cmd_dim
        self.num_primitives = num_primitives
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        # activation
        if activation == "elu":
            act = nn.ELU
        else:
            act = nn.ReLU

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        # -----------------------
        # Compatibility matrix (learnable)
        # -----------------------
        self.compat_matrix = nn.Parameter(torch.zeros(cmd_dim, cmd_dim))
        with torch.no_grad():
            self.compat_matrix.fill_(0.5)   # moderate compatibility
            for i in range(cmd_dim):
                self.compat_matrix[i,i] = 2.0  # prefer self-support

        # -----------------------
        # command projector
        # -----------------------
        self.command_projector = nn.Sequential(
            nn.Linear(cmd_dim, 64),
            act(),
            nn.Linear(64, num_primitives)
        )

        # -----------------------
        # FiLM proj
        # -----------------------
        self.film_gamma = nn.Linear(primitive_dim, obs_latent_dim)
        self.film_beta = nn.Linear(primitive_dim, obs_latent_dim)

        # -----------------------
        # primitive bank
        # -----------------------
        self.primitive_dim = primitive_dim
        self.primitive_bank = nn.Parameter(torch.randn(num_primitives, primitive_dim) * 0.1)

        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpBarlowTwinsActorUnified(num_prop=num_prop,
                                      num_hist=10,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[256,128])
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32+ primitive_dim,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32+ primitive_dim,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ===========================
    # compatibility_mask (improved)
    # ===========================
    def compatibility_mask(self, cmd, binary_primitives=None,
                        eps=1e-8, hard_threshold=0.5):
        """
        Compatibility mask that:
        * preserves continuous cmd semantics
        * threshold only binary primitives
        * uses compatibility matrix to generate multiplicative weights
        Inputs:
            cmd: [B,K]  raw commands (continuous or binary)
            binary_primitives: list[bool] length K
        Returns:
            cmd_masked: [B,K]
            compat_factor: [B,K] in (0,1]
        """
        B, K = cmd.shape
        device = cmd.device

        # -----------------------------
        # 1) Process binary primitives
        #    (DO NOT normalize continuous commands)
        # -----------------------------
        if binary_primitives is not None:
            bin_mask = torch.tensor(binary_primitives,
                                    dtype=torch.bool, device=device)
            if bin_mask.numel() != K:
                raise ValueError("binary_primitives must have length K")

            # clone to avoid modifying original cmd
            cmd_proc = cmd.clone()

            # only apply threshold to binary positions
            # cmd >= 0.5 → 1   else 0
            cmd_proc[:, bin_mask] = (cmd_proc[:, bin_mask] >= hard_threshold).float()

        else:
            # no binary primitives → keep as-is
            cmd_proc = cmd.clone()

        # -----------------------------
        # 2) Build compatibility matrix C ∈ (0,1)
        # -----------------------------
        C = torch.sigmoid(self.compat_matrix)  # learnable
        C = 0.5 * (C + C.t())                  # symmetric

        # -----------------------------
        # 3) Compute compatibility factor
        #    compat_factor[b,i] = Π_j C[j,i]^(cmd_proc[b,j])
        #    Use log-sum-exp trick for stability
        # -----------------------------
        # logC[j,i]
        logC = torch.log(C + eps)  # [K,K]

        # log_score[b,i] = Σ_j cmd_proc[b,j] * logC[j,i]
        log_score = torch.matmul(cmd_proc, logC)  # [B,K]

        compat_factor = torch.exp(log_score)      # [B,K]
        compat_factor = torch.clamp(compat_factor, 0.0, 1.0)

        # -----------------------------
        # 4) Apply multiplicative mask
        # -----------------------------
        cmd_masked = cmd * compat_factor

        return cmd_masked, compat_factor

    # ===========================
    # compose_skill (uses compatibility_mask)
    # ===========================
    def compose_skill(
        self,
        cmd,
        primitive_mask=None,
        binary_primitives=None,
        eps=1e-6,
        softplus_beta=1.0        # 控制 softplus 平滑度
    ):
        """
        最稳定的技能组合方式：
        - 无需 softmax / 非softmax 两种模式；
        - 永远不会出 NaN;
        - 权重始终可导；
        - 权重永远非零；
        - primitive bank 可在 RL 中稳定学习。
        """

        if cmd is None:
            raise ValueError("cmd must be provided to compose_skill")

        # 1. compatibility-aware scaling
        cmd_masked, compat_factor = self.compatibility_mask(cmd, binary_primitives=binary_primitives)

        # 2. projector 输出
        proj = self.command_projector(cmd_masked)  # [B,K]

        if primitive_mask is not None:
            # primitive_mask 可为 0/1 或 0..1 连续值
            # 保证类型和 device
            primitive_mask = primitive_mask.to(proj.device).type_as(proj)
            if primitive_mask.shape != proj.shape:
                raise ValueError("primitive_mask must have shape [B, K]")
            proj = proj * primitive_mask

        # 3. Softplus 生成 strictly-positive 权重，防止 w=0
        #    softplus_beta 参数可控制 sharpness（beta 越大越像 ReLU）
        w_raw = F.softplus(proj, beta=softplus_beta) + eps   # [B,K]

        # 4. Normalize → 保证 sum(w)=1，始终可导
        w = w_raw / (w_raw.sum(dim=1, keepdim=True) + eps)

        # 5. 合成 primitive latent
        z = torch.matmul(w, self.primitive_bank)   # [B, D]
        z = F.layer_norm(z, [z.shape[1]])

        return z, w, compat_factor

    def build_primitive_mask_from_cmd(cmd):
        """
        cmd: torch.Tensor, shape [B, 8]
        return: primitive_mask: torch.Tensor, shape [B, K], values 0/1
                info dict with selected gait per-sample for debug
        """
        PR_WALK   = 0
        PR_WHEEL  = 1
        PR_JUMP   = 2
        PR_SPIN   = 3
        PR_HEIGHT = 4
        K = 5
        B = cmd.shape[0]
        device = cmd.device

        # extract flags (容错：允许 float 形式的 0/1 或 连续值)
        gait_mode = (cmd[:, 5] > 0.5).to(torch.float32)   # 1 = WALK, 0 = WHEEL
        jump_flag = (cmd[:, 6] > 0.5).to(torch.float32)   # 1 = JUMP requested
        spin_flag = (cmd[:, 7] > 0.5).to(torch.float32)   # 1 = SPIN requested

        primitive_mask = torch.zeros(B, K, device=device, dtype=torch.float32)

        # 1) Determine base gait per-sample (JUMP overrides gait_mode)
        #    gait_choice: 0=WHEEL, 1=WALK, 2=JUMP  (we'll map to primitives)
        # Note: we use boolean masks for batch operations
        jump_env = jump_flag.bool()
        walk_env = (~jump_env) & (gait_mode.bool())
        wheel_env = (~jump_env) & (~gait_mode.bool())

        # assign base gait primitive
        primitive_mask[walk_env, PR_WALK]   = 1.0
        primitive_mask[wheel_env, PR_WHEEL] = 1.0
        primitive_mask[jump_env, PR_JUMP]   = 1.0

        # 2) Handle SPIN: only allowed when base gait == WHEEL and spin_flag==1
        spin_allowed_env = wheel_env & spin_flag.bool()
        primitive_mask[spin_allowed_env, PR_SPIN] = 1.0

        # 3) Handle HEIGHT: allowed if base gait is WALK or WHEEL AND not (jump active) AND not (spin active and spin_allowed)
        #    i.e., height disabled during JUMP or during a valid SPIN.
        #    note: spin_flag might be 1 but spin_allowed_env false (when gait is WALK) -> then spin not active.
        #    height_enabled if (walk_env or wheel_env) AND NOT jump_env AND NOT spin_allowed_env
        height_allowed_env = (walk_env | wheel_env) & (~jump_env) & (~spin_allowed_env)
        primitive_mask[height_allowed_env, PR_HEIGHT] = 1.0

        # debug info
        info = {
            "jump_env": jump_env,
            "walk_env": walk_env,
            "wheel_env": wheel_env,
            "spin_allowed_env": spin_allowed_env,
            "height_allowed_env": height_allowed_env
        }

        return primitive_mask, info

    def act_teacher(self,obs, **kwargs):
        B = obs.shape[0]
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)

        cmd = obs_prop[:,6:13]
        primitive_mask,_ = self.build_primitive_mask_from_cmd(cmd)

        binary_primitives = [False, False, False, False, True, True, True]
        # compose skill latent
        z_skill, w, compat_factor = self.compose_skill(cmd, primitive_mask, binary_primitives=binary_primitives, **kwargs)

        # obs latent
        obs_latent = self.actor_teacher_backbone.obs_encoder(obs_prop).detach()  # [B, obs_latent_dim]

        # FiLM
        gamma = self.film_gamma(z_skill)
        beta = self.film_beta(z_skill)
        obs_latent_mod = gamma * obs_latent + beta

        mean = self.actor_teacher_backbone.forward(obs_prop, obs_hist, obs_latent_mod)

        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        cmd = obs_prop[:,6:13]
        binary_primitives = [False, False, False, False, True, True, True]
        z_skill, _, _ = self.compose_skill(cmd, binary_primitives=binary_primitives)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent,z_skill], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        cmd = obs_prop[:,6:13]
        binary_primitives = [False, False, False, False, True, True, True]
        z_skill, _, _ = self.compose_skill(cmd, binary_primitives=binary_primitives)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent,z_skill], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        # priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 3]#只估线速度
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 10]#只估线速度
        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self, path, device):
        obs_demo_input = torch.randn(1, self.num_prop, device=device)
        hist_demo_input = torch.randn(1, self.num_hist, self.num_prop, device=device)

        # 保存 TorchScript
        model_jit = torch.jit.trace(self.actor_teacher_backbone, (obs_demo_input, hist_demo_input))
        model_jit.save(path)
        print(f"[✓] TorchScript policy saved to {path}")

        # 导出 ONNX，固定输入输出名字
        torch.onnx.export(
            self.actor_teacher_backbone,
            (obs_demo_input, hist_demo_input),
            "test.onnx",
            verbose=True,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["obs", "obs_hist"],   # 👈 改为固定名字
            output_names=["action"],           # 👈 改为固定名字
            dynamic_axes=None                  # 固定 batch=1
        )
        print(f"[✓] ONNX policy saved to test.onnx")

class MlpBarlowTwinsActorUnified(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBarlowTwinsActorUnified,self).__init__()
        self.mlp_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                #  out_dims=latent_dim+3,
                                out_dims=latent_dim+10,
                                 hidden_dims=mlp_encoder_dims))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                #  input_dims=latent_dim + num_prop + 3,
                                 input_dims=latent_dim + num_prop + 10 + latent_dim,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist,obs_latent_mod):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        obs_hist_full = obs_hist_full[:,0:,:].view(b,-1)
        latents = self.mlp_encoder(obs_hist_full)
        actor_input = torch.cat([latents,obs,obs_latent_mod],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        obs_hist = obs_hist[:,0:,:].view(b,-1)
        predicted = self.mlp_encoder(obs_hist)
        # hist_latent = predicted[:,3:]
        # priv_latent = predicted[:,:3]
        hist_latent = predicted[:,10:]
        priv_latent = predicted[:,:10]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss,priv_loss