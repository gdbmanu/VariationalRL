#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal


# In[2]:


from copy import deepcopy
import gym
from gym import spaces
import time
from spinup.utils.logx import EpochLogger


# In[3]:


import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# ### V: large footprint (1000000) replay buffer
# ### moderate batch_size (100)
# 
# #### zh : policy sampling tests : ref policy sampling on Q_test for initial training (1e5 first interactions) 
# #### zl : changing self.test_polyak
# #### zm : update simplification
# #### zp : back to original update without Q_KL
# #### zq : 0.5 in KL loss term
# #### zs : sampling improvement

# #### Hyperparameters

# In[4]:


env_fn = lambda : gym.make('BipedalWalker-v3')


# In[5]:


BETA_REF = 30 # reward amplification
k = 10 # explo/exploit balance
PREC = k / 10
E3D = True # !!!
do_reward=True


# In[6]:


render = False


# In[7]:


gamma = 0.99


# In[8]:


start_steps=10000
steps_per_epoch=4000
update_after=10000
update_every=50
epochs=500
replay_size = 1000000 #int(1e6)
batch_size = 100


# In[9]:


output_dir=f'e3d-sac-walker-BETA={BETA_REF}-k={k}-E3D={E3D}-zs'
exp_name=f'e3d-sac-walker-BETA={BETA_REF}-k={k}-E3D={E3D}-zs'


# ## Spinup utilities

# In[10]:


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# In[11]:


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


# In[12]:


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# ## Neural networks

# In[13]:


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# In[14]:


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


# In[15]:


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


# In[16]:


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q_prim = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


# In[17]:


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ep_len_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, ep_len):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ep_len_buf[self.ptr] = ep_len
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     ep_len=self.ep_len_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def last_obs_2(self, batch_size=32, batch_interval=1000):
        
        if self.size < batch_interval:
            batch_interval = self.size
        start=self.ptr-batch_interval
        
        if start>=0:
            path_slice = slice(start, self.ptr)
            last_obs_2_interval = self.obs2_buf[path_slice]
        else:
            slice_1 = slice(self.max_size+start, self.max_size)
            slice_2 = slice(0, self.ptr)
            last_obs_2_interval = np.concatenate((self.obs2_buf[slice_1],self.obs2_buf[slice_2]))
            
        idxs = np.random.randint(0, batch_interval, size=batch_size)
        last_obs_2_batch=last_obs_2_interval[idxs]
        
        return torch.as_tensor(last_obs_2_batch, dtype=torch.float32) 


# ## Trainer class

# In[18]:


from scipy.special import gamma as gamma_fn

class KNN_prob():
    def __init__(self, data, k=10):
        self.data = data
        self.n, self.d = self.data.shape
        self.V = np.pi**(self.d/2) / gamma_fn(self.d/2 + 1)
        self.k = k
        self.q = k/self.n        
        #print(self.n, self.d, self.q)
    def __call__(self, x):
        dists = np.sqrt(np.sum((x - self.data)**2,1))
        dist = np.quantile(dists, self.q)
        return self.k/ (self.n * self.V * dist**self.d)


# In[19]:


class SAC_Trainer:
    
    def __init__(self, env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, E3D=True, do_reward=True):
        
        """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """  

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        self.gamma = gamma
        self.polyak = polyak
        '''self.KL_polyak = 1 - 3 * (1 - self.polyak)
        self.ref_polyak = 1 - 3 * (1 - self.polyak)'''
        self.test_polyak = polyak #1 - 3 * (1 - self.polyak) # !!! zl !!!
        self.lr = lr
        self.alpha = alpha
        
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.actor = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.actor_targ = deepcopy(self.actor)
        
        self.actor_test = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.actor_test_targ = deepcopy(self.actor_test)
        
        '''self.q_KL = MLPQFunction(self.env.observation_space.shape[0], 
                                 self.env.action_space.shape[0], 
                                 **ac_kwargs)       
        self.q_KL_targ = deepcopy(self.q_KL)
        
        self.q_ref = MLPQFunction(self.env.observation_space.shape[0], 
                                 self.env.action_space.shape[0], 
                                 **ac_kwargs)       
        self.q_ref_targ = deepcopy(self.q_ref)'''

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.actor_test_targ.parameters():
            p.requires_grad = False
        '''for p in self.q_KL_targ.parameters():
            p.requires_grad = False
        for p in self.q_ref_targ.parameters():
            p.requires_grad = False'''


        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.actor.pi, self.actor.q])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)        
        
        self.batch_size=batch_size
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.actor.q.parameters(), lr=self.lr)
        self.pi_test_optimizer = Adam(self.actor_test.pi.parameters(), lr=self.lr)
        self.q_test_optimizer = Adam(self.actor_test.q.parameters(), lr=self.lr)
        '''self.q_KL_optimizer = Adam(self.q_KL.parameters(), lr=self.lr)
        self.q_ref_optimizer = Adam(self.q_ref.parameters(), lr=self.lr)'''

        # Set up model saving
        self.logger.setup_pytorch_saver(self.actor)
        self.logger.setup_pytorch_saver(self.actor_test)
        '''self.logger.setup_pytorch_saver(self.q_KL)
        self.logger.setup_pytorch_saver(self.q_ref)'''
        
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.save_freq = save_freq
        
        self.E3D = E3D
        self.do_reward = do_reward

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        s, a, r, s_prim, done, ep_len = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['ep_len']
       
        #q_KL = self.q_KL(s,a) # to be updated
        #q_ref = self.q_ref(s,a) # to be updated        
        q = self.actor.q(s,a) # to be updated
        
        # Bellman backup for Q function
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prim, logp_a_prim = self.actor.pi(s_prim)     
            #a_prim_ref, logp_a_prim_ref = self.actor.pi(s_prim)  
            
            # (uniform) KL drive
            if self.E3D:               
                p_vect = np.vectorize(self.state_probs, signature='(m)->()')
                p_obs_np = p_vect(s_prim.detach().numpy())
                p_obs = torch.as_tensor(p_obs_np, dtype=torch.float32)
                log_p_obs = torch.log(p_obs)
                log_p_obs = torch.clamp(log_p_obs, -50, 50)                
                log_p_info = dict(LogPObs=log_p_obs.detach().numpy())
                log_p_obs_c = log_p_obs - torch.mean(log_p_obs)
                # !!
                TD_err =  - (1 - self.gamma)/BETA_REF * log_p_obs_c
                done_KL = torch.max(done, torch.as_tensor(ep_len == self.max_ep_len, dtype=torch.float32))
            else:
                log_p_info = dict(LogPObs=None)
                
            LAMBDA = PREC
                        
            # Target Q-value
            if not self.do_reward:
                r = 0
                
            q_pi_targ = self.actor_targ.q(s_prim, a_prim)
            if self.E3D:    
                backup_ref =  r +  (1 - done) * self.gamma * q_pi_targ 
                backup_KL  = self.actor.q(s,a) + (1 - done_KL) * TD_err
            else:
                backup =  r + self.gamma * (1 - done) * (q_pi_targ - 1 / BETA_REF * logp_a_prim) # SAC update
            #q_pi_ref_targ = self.q_ref_targ(s_prim, a_prim_ref) #!! q_ref for q_ref update !!
            #backup_ref =  r + self.gamma * (1 - done) * q_pi_ref_targ  

        # MSE loss against Bellman backup  
                
        if self.E3D:  
            loss_q =  0.5 * ((q - backup_KL)**2).mean() + 0.5 * LAMBDA * ((q - backup_ref)**2).mean() 
        else:
            loss_q = 0.5 * ((q - backup)**2).mean() 

        # Useful info for logging
        #q_KL_info = dict(QKLVals=q_KL.detach().numpy())
        #q_ref_info = dict(QRefVals=q_ref.detach().numpy())
        q_info = dict(QVals=q.detach().numpy())
        
        return loss_q,  q_info, log_p_info

    def compute_loss_q_test(self, data):
        s, a, r, s_prim, done, ep_len = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['ep_len']
         
        q_test = self.actor_test.q(s,a) # to be updated
        
        # Bellman backup for Q function
        with torch.no_grad(): 
            if self.t > 1e5:   # !!! zh !!!
                a_prim_test, logp_a_prim_test = self.actor_test.pi(s_prim)  
            else:
                a_prim_test, logp_a_prim_test = self.actor.pi(s_prim)  
            
            LAMBDA = PREC            
            
            # Target Q-value
            if not self.do_reward:
                r = 0
                
            q_pi_test_targ_1 = self.actor_test_targ.q(s_prim, a_prim_test) # !! zl2
            q_pi_test_targ_2 = self.actor_targ.q(s_prim, a_prim_test)
            q_pi_test_targ = torch.min(q_pi_test_targ_1, q_pi_test_targ_2)
            backup_test =  r + self.gamma * (1 - done) * q_pi_test_targ 
            

        # MSE loss against Bellman backup  
                
        loss_q_test = 0.5 * ((q_test - backup_test)**2).mean()

        # Useful info for logging
        q_test_info = dict(QTestVals=q_test.detach().numpy())
        
        return loss_q_test, q_test_info    
    
    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        s = data['obs']
        #pi, logp_pi = ac.pi(s)
        a, logp_a = self.actor.pi(s)
        q_pi = self.actor.q(s, a)
        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_a - BETA_REF * q_pi).mean()
        # Useful info for logging
        pi_info = dict(LogPi=logp_a.detach().numpy())

        return loss_pi, pi_info
    
    # Set up function for computing SAC pi loss
    def compute_loss_pi_test(self, data):
        s = data['obs']
        #pi, logp_pi = ac.pi(s)
        a, logp_a = self.actor_test.pi(s)
        
        q1_pi = self.actor_test.q(s, a)
        q2_pi = self.actor.q(s, a)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        loss_pi_test = (self.alpha * logp_a - BETA_REF * q_pi).mean()
        # Useful info for logging
        pi_test_info = dict(LogPiTest=logp_a.detach().numpy())

        return loss_pi_test, pi_test_info

    def update(self, data):
        # First run one gradient descent step for Q.
        loss_q, q_info, log_p_info = self.compute_loss_q(data)
            
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()        
            
        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info, 
                          **log_p_info)

        
        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.actor.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.actor.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            
    def update_test(self, data):
        # First run one gradient descent step for Q.
        loss_q_test, q_test_info = self.compute_loss_q_test(data)
                           
        self.q_test_optimizer.zero_grad()
        loss_q_test.backward()
        self.q_test_optimizer.step()                
            
        # Record things
        self.logger.store(LossQTest=loss_q_test.item(), **q_test_info)
        
        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.actor_test.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_test_optimizer.zero_grad()
        loss_pi_test, pi_test_info = self.compute_loss_pi_test(data)
        loss_pi_test.backward()
        self.pi_test_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.actor_test.q.parameters():
            p.requires_grad = True        

        # Record things
        self.logger.store(LossPiTest=loss_pi_test.item(), **pi_test_info)        

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_test.parameters(), self.actor_test_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.test_polyak)
                p_targ.data.add_((1 - self.test_polyak) * p.data)
                
                
    def get_action(self, obs, test=False, deterministic=False):
        if test:
            return self.actor_test.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)
        else:
            return self.actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            obs, done, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(done or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                obs, r, done, _ = self.test_env.step(self.get_action(obs, test=True, deterministic=False))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train(self, init=True):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        s, ep_ret, ep_len = self.env.reset(), 0, 0
        
        if init:
            self.t = 0

        # Main loop: collect experience in env and update/log each epoch
        while self.t < total_steps:

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if self.t > self.start_steps:
                a = self.get_action(s)
            else:
                a = self.env.action_space.sample()

            # Step the env
            s_prim, r, done, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if ep_len==self.max_ep_len else done

            # Store experience to replay buffer
            self.replay_buffer.store(s, a, r, s_prim, done, ep_len)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            s = s_prim

            # End of trajectory handling
            if done or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                s, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if self.t >= self.update_after and self.t % self.update_every == 0:                
                for _ in range(self.update_every):          
                    state_batch_size = min(self.replay_buffer.size, 1000)   
                    #state_batch = self.replay_buffer.last_obs_2(batch_size=state_batch_size, 
                    #                                            batch_interval=100000)
                    state_batch = self.replay_buffer.sample_batch(state_batch_size)['obs2']
                    self.state_probs = KNN_prob(state_batch.detach().numpy())
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)
                    batch_test = self.replay_buffer.sample_batch(self.batch_size)
                    self.update_test(data=batch_test)

            # End of epoch handling
            if (self.t+1) % self.steps_per_epoch == 0:
                epoch = (self.t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                if self.t >= self.update_after:
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('EpRet', with_min_and_max=True)
                    self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('TestEpLen', average_only=True)
                    self.logger.log_tabular('TotalEnvInteracts', self.t)
                    self.logger.log_tabular('LogPObs', with_min_and_max=True)
                    self.logger.log_tabular('QVals', with_min_and_max=True)
                    self.logger.log_tabular('QTestVals', with_min_and_max=True)
                    self.logger.log_tabular('LogPi', average_only=True)   
                    self.logger.log_tabular('LogPiTest', average_only=True)  
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossPiTest', average_only=True)
                    self.logger.log_tabular('LossQ', average_only=True)
                    self.logger.log_tabular('LossQTest', average_only=True)                
                    self.logger.log_tabular('Time', time.time()-start_time)
                    self.logger.dump_tabular()
                    
            self.t += 1


# ## Training

# In[20]:


ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU) #activation=nn.Tanh) #

logger_kwargs = dict(output_dir=output_dir, exp_name=exp_name)

trainer = SAC_Trainer(env_fn=env_fn, 
                       ac_kwargs=ac_kwargs, 
                       start_steps=start_steps, 
                       steps_per_epoch=steps_per_epoch, 
                       update_after=update_after, 
                       update_every=update_every,
                       epochs=epochs, 
                       logger_kwargs=logger_kwargs,
                       gamma=gamma,
                       alpha=1,
                       replay_size = replay_size,
                       E3D=E3D,
                       do_reward=do_reward,
                       batch_size=batch_size)


# In[21]:


trainer.train()


# In[ ]:


trainer.E3D


# In[22]:


plt.plot(np.cumsum(trainer.replay_buffer.rew_buf))
plt.plot(trainer.replay_buffer.ptr,0,'r.',markersize=10)


# In[23]:


inter = slice(7000,10000)
plt.figure(figsize=(20,5))
obs = torch.as_tensor(trainer.replay_buffer.obs_buf[inter], dtype=torch.float32)
act = torch.as_tensor(trainer.replay_buffer.act_buf[inter], dtype=torch.float32)
plt.plot(trainer.actor.q(obs, act).detach().numpy(),'g', label='Q')
plt.plot(trainer.replay_buffer.done_buf[inter]*100,'r')
plt.plot(trainer.replay_buffer.rew_buf[inter],label='rew')

plt.legend()
plt.figure(figsize=(20,5))
plt.plot(trainer.replay_buffer.act_buf[inter])
plt.plot(trainer.replay_buffer.done_buf[inter],'r')


# In[24]:


inter = slice(trainer.replay_buffer.ptr-3000,trainer.replay_buffer.ptr)
plt.figure(figsize=(20,5))
obs = torch.as_tensor(trainer.replay_buffer.obs_buf[inter], dtype=torch.float32)
act = torch.as_tensor(trainer.replay_buffer.act_buf[inter], dtype=torch.float32)
plt.plot(trainer.actor.q(obs, act).detach().numpy(),'g', label='Q')
plt.plot(trainer.replay_buffer.done_buf[inter]*100,'r')
plt.plot(trainer.replay_buffer.rew_buf[inter],label='rew')

plt.legend()
plt.figure(figsize=(20,5))
plt.plot(trainer.replay_buffer.act_buf[inter])
plt.plot(trainer.replay_buffer.done_buf[inter],'r')


# ## Animation

# In[26]:


render=True
if render:
    env =  trainer.env
    obs = env.reset()
    done=False
    t=0
    ret=0
    while not done:
        t+=1
        env.render()
        a = trainer.get_action(obs, test=True, deterministic=False)
        
        obs, r, done, _ = env.step(a)
        obs_t, a_t = torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)
        print(f'''t 
                 Q: {trainer.actor.q(obs_t.unsqueeze(0), a_t.unsqueeze(0)).detach().numpy()[0]}, 
                 logP:{np.log(trainer.state_probs(obs))}, 
                 reward: {r}''')
        ret+= r
print(f'EpLen : {t}, total return: {ret}')


# In[27]:


import pandas


# In[28]:


mat = pandas.read_csv(output_dir+'/progress.txt','\t')


# In[29]:


mat.keys()


# In[30]:


for metric in ['LogPObs', 'EpRet', 'TestEpRet']:
    plt.figure(figsize=(10,5))
    plt.plot(mat.TotalEnvInteracts, mat[f'Average{metric}'])
    plt.fill_between(mat.TotalEnvInteracts, mat[f'Average{metric}']-mat[f'Std{metric}'], 
                     mat[f'Average{metric}']+mat[f'Std{metric}'],alpha=.3)
    plt.plot(mat.TotalEnvInteracts,  mat[f'Max{metric}'],'.b',markersize=4, alpha=.3)
    plt.plot(mat.TotalEnvInteracts,  mat[f'Min{metric}'],'.b',markersize=4, alpha=.3)
    plt.title(metric)
    plt.xlabel('#Epochs')
    plt.xlim(0,2e6)
plt.figure(figsize=(10,5))
for metric, color in zip([ 'QVals', 'QTestVals'], [ 'blue', 'orange', 'green']):
    plt.plot(mat[f'Average{metric}'], label=metric, color=color)
    plt.xlim(0,500)
plt.legend()
plt.figure(figsize=(10,5))
for metric, color in zip([ 'LogPi', 'LogPiTest'], ['orange', 'green']):
    plt.plot(mat[f'{metric}'], label=metric, color=color)
    plt.xlim(0,500)
plt.legend()
plt.title('LogPi, LogPiTest')
plt.figure(figsize=(10,5))
for metric, color in zip([ 'LossPi', 'LossPiTest'], ['orange', 'green']):
    plt.plot(mat[f'{metric}'], label=metric, color=color)
    plt.xlim(0,500)
plt.legend()
plt.title('LossPi, LossPiTest')

for metric in['EpLen', 'TestEpLen']:
    plt.figure(figsize=(10,5))
    plt.plot(mat[f'{metric}'])
    plt.title(metric)
    plt.xlabel('#Epochs')
    plt.xlim(0,500)
    
plt.figure(figsize=(10,5))
for metric, color in zip([ 'LossQ', 'LossQTest'], ['orange', 'green']):
    plt.plot(mat[f'{metric}'], label=metric, color=color)
    plt.xlim(0,500)
plt.legend()
plt.title('LossQRef, LossQ, LossQTest')


# ## Action space random sampling

# In[ ]:


mem_obs_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
mem_reward_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
mem_act_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}

n_sample = 1000
for num_sample in range(n_sample):    
    obs = env.reset()
    #action = env.action_space.sample() #np.random.rand(2) * 2 - 1
    for step in range(5):
        action = trainer.get_action(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)
        mem_obs_dict[step].append(obs)
        mem_act_dict[step].append(action)
        mem_reward_dict[step].append(reward)
    if num_sample %50 == 0:
        print(f'num_sample: {num_sample}, obs={obs}, reward={reward}')


# ## Distribution of final position

# In[ ]:


for step in range(5):
    mem_obs=np.array(mem_obs_dict[step])
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-1.5, 2.5), ylim=(-2, 2))
    ax.grid()
    #line, = ax.plot([], [], 'o-', lw=2)
    #line.set_data(*env.doubleJointArm.position())
    plt.scatter(mem_obs[:,2], mem_obs[:,3], alpha = .1, c='green')


# In[ ]:


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return 


# In[ ]:


for step in range(5):
    obs_dict={}
    for i, obs in enumerate(mem_obs_dict[step]):
        key = tuple(obs[:2])
        if key not in obs_dict:
            obs_dict[key] = [mem_obs_dict[step][i]]
        else:
            obs_dict[key].append(mem_obs_dict[step][i])
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1.5, 2.5), ylim=(-2, 2))
    ax.grid()

    color = {}
    for i, key in enumerate(obs_dict.keys()):
        c = np.array(plt.cm.get_cmap('rainbow', 6)(i))
        c = np.expand_dims(c,0)
        color[key] = c

    cpt = 0
    for key in obs_dict:
        obs_dict[key] = np.array(obs_dict[key])
        plt.scatter(obs_dict[key][:,2], obs_dict[key][:,3], alpha = .1, label=cpt, c=color[key])
    for key in obs_dict:
        obs_dict[key] = np.array(obs_dict[key])
        plt.scatter(obs_dict[key][:,0], obs_dict[key][:,1], c=color[key])

ax = np.linspace(-2,3,100)
data = np.concatenate((np.zeros((100,2)), np.expand_dims(ax, 1), -0.5*np.ones((100,1)), np.zeros((100,2))), axis=1)
for k in (5, 8, 10,12, 15): #, 20, 30, 50):
    p = KNN_prob(mem_obs, k=k)
    p_vect = np.vectorize(p, signature='(m)->()')
    f = p_vect(data)
    plt.plot(ax, f, label=k)
plt.legend()
# In[ ]:


for step in range(5):
    mem_act=np.array(mem_act_dict[step])
    fig = plt.figure(figsize=(5,5))
    plt.scatter(mem_act[:,0], mem_act[:,1], alpha = .1, c='orange')
    plt.xlim([-1,1])
    plt.ylim([-1,1])


# In[ ]:


act_dict={}
for i, obs in enumerate(mem_obs):
    key = tuple(obs[:2])
    if key not in act_dict:
        act_dict[key] = [mem_act[i]]
    else:
        act_dict[key].append(mem_act[i])
fig = plt.figure(figsize=(8,8))
plt.xlim([-1,1])
plt.ylim([-1,1])
cpt = 0
for i, key in enumerate(act_dict.keys()):
    cpt += 1
    act_dict[key] = np.array(act_dict[key])
    plt.scatter(act_dict[key][:,0], act_dict[key][:,1], alpha = .1, c=color[key])
#plt.legend()


# In[ ]:


trainer.polyak, 1 - 3 * (1 - trainer.polyak)


# In[ ]:


env=env_fn()
env.observation_space.shape, env.action_space.shape


# In[ ]:




