from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import gym
import time
import spinup.algos.pytorch.macao.core as core
from spinup.utils.logx import EpochLogger
from sklearn.neighbors import KernelDensity


from scipy.special import softmax

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ep_len_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp, next_obs, done, ep_len):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
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
                     logp=self.logp_buf[idxs],
                     done=self.done_buf[idxs],
                     ep_len=self.ep_len_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def importance_sample_batch(self, batch_size=32, sel_range=3):    
        idxs_sel = np.random.randint(0, self.size, size=batch_size*sel_range)
        logp_scores = self.logp_buf[idxs_sel].reshape(batch_size, sel_range)
        idxs_sel = idxs_sel.reshape(batch_size, sel_range)
        idxs = np.zeros(batch_size).astype('int')
        for i_batch in range(batch_size):
            i_softmax = np.random.choice(sel_range, 1, p=softmax(-logp_scores[i_batch,:]))[0]
            idxs[i_batch] = idxs_sel[i_batch,i_softmax]
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     logp=self.logp_buf[idxs],
                     done=self.done_buf[idxs],
                     ep_len=self.ep_len_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def last_batch(self, batch_size=32, batch_interval=1000):
        
        assert batch_size <= batch_interval
        
        if self.size < batch_interval:
            batch_interval = self.size
        start=self.ptr-batch_interval
        end = self.ptr-batch_interval + batch_size
        print(start, end)
        
        if start>=0 or (start < 0 and end < 0):
            if end < 0:
                start = self.max_size + start
            path_slice = slice(start, start+batch_size) #self.ptr)
            print(path_slice)
            batch = dict(obs=self.obs_buf[path_slice],
                         obs2=self.obs2_buf[path_slice],
                         act=self.act_buf[path_slice],
                         rew=self.rew_buf[path_slice],
                         logp=self.logp_buf[path_slice],
                         done=self.done_buf[path_slice],
                         ep_len=self.ep_len_buf[path_slice])
        else:
            slice_1 = slice(self.max_size+start, self.max_size)
            slice_2 = slice(0, end) #self.ptr)
            batch = dict(obs=np.concatenate((self.obs_buf[slice_1],self.obs_buf[slice_2])),
                         obs2=np.concatenate((self.obs2_buf[slice_1],self.obs2_buf[slice_2])),
                         act=np.concatenate((self.act_buf[slice_1],self.act_buf[slice_2])),
                         rew=np.concatenate((self.rew_buf[slice_1],self.rew_buf[slice_2])),
                         logp=np.concatenate((self.logp_buf[slice_1],self.logp_buf[slice_2])),
                         done=np.concatenate((self.done_buf[slice_1],self.done_buf[slice_2])),
                         ep_len=np.concatenate((self.ep_len_buf[slice_1],self.ep_len_buf[slice_2])))        
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
class CCA_Trainer:
    
    def __init__(self, env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, beta=10, prec=0.3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, bandwidth=0.1, algo='cca'):
                
        """
    Concurrent Credit Assignment (CCA)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q``, and ``q`` should return:

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

        beta (float): Reward scale
        
        prec (float) : Reward precision (lambda)

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
        self.lr = lr
        self.beta = beta
        self.prec= prec
        
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.actor = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.actor_targ = deepcopy(self.actor)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.actor.pi, self.actor.pi_test,
                                                             self.actor.q_1, self.actor.q_2, 
                                                             self.actor.q_ref_1, self.actor.q_ref_2,
                                                             self.actor.q_KL_1, self.actor.q_KL_2])
        self.logger.log('\nNumber of parameters: \t pi: %d %d, \t q: %d, %d, %d, %d, %d, %d\n'%var_counts)        
        
        self.batch_size = batch_size
        
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.actor.q_1.parameters(), self.actor.q_2.parameters(), 
                                        self.actor.q_ref_1.parameters(), self.actor.q_ref_2.parameters(),
                                        self.actor.q_KL_1.parameters(), self.actor.q_KL_2.parameters())
        
        self.pi_params = itertools.chain(self.actor.pi.parameters(), self.actor.pi_test.parameters()) 
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.actor)
        
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.save_freq = save_freq
        
        self.bandwidth=bandwidth
        self.algo=algo
        
    
    INV_TMP_SOFTMIN = 1
    def softmin(self, q1, q2):
        q_cat = torch.cat((q1.unsqueeze(1), 
                           q2.unsqueeze(1)), 
                           dim=1)
        with torch.no_grad():
            p_q = F.softmin(INV_TMP_SOFTMIN * q_cat, dim=1) 
            #p_q = torch.clamp(p_q, 0, 1)           
            i_q = Categorical(p_q).sample().detach().numpy()
            idx = np.array((np.arange(self.batch_size), i_q))
        return q_cat[idx]
        

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        s, a, r, s_prim, done, ep_len = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['ep_len']
       
        q_1 = self.actor.q_1(s,a) # to be updated
        q_2 = self.actor.q_2(s,a) # to be updated
        
        if self.algo == 'cca':   
			q_ref_1 = self.actor.q_ref_1(s,a) # to be updated
			q_ref_2 = self.actor.q_ref_2(s,a) # to be updated
			
			q_KL_1 = self.actor.q_KL_1(s,a) # to be updated
			q_KL_2 = self.actor.q_KL_2(s,a) # to be updated
        
        # Bellman backup for Q function
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prim_1, logp_a_prim_1 = self.actor.pi(s_prim)     
            a_prim_2, logp_a_prim_2 = self.actor.pi(s_prim)     
            
            # (uniform) KL drive
            if self.algo == 'cca':          
                log_p_obs_np = self.state_probs.score_samples(s_prim.detach().numpy())
                log_p_obs = torch.as_tensor(log_p_obs_np, dtype=torch.float32)
                log_p_info = dict(LogPObs=log_p_obs.detach().numpy())
                log_p_obs_c = log_p_obs - torch.mean(log_p_obs)
                TD_err =  - 1 / self.beta * log_p_obs_c
                done_KL = torch.max(done, torch.as_tensor(ep_len == self.max_ep_len, dtype=torch.float32))
            else:
                log_p_info = dict(LogPObs=None)
                            
            # Target Q-values
            q1_targ = self.actor_targ.q_1(s_prim, a_prim_1)
            q2_targ = self.actor_targ.q_2(s_prim, a_prim_2)
            q_targ = self.softmin(q1_targ, q2_targ)
            
            if self.algo == 'cca':    
				q1_ref_targ = self.actor_targ.q_ref_1(s_prim, a_prim_1)
				q2_ref_targ = self.actor_targ.q_ref_2(s_prim, a_prim_2)
				q_ref_targ = self.softmin(q1_ref_targ, q2_ref_targ)
						   
				q1_KL_targ = self.actor_targ.q_KL_1(s_prim, a_prim_1)
				q2_KL_targ = self.actor_targ.q_KL_2(s_prim, a_prim_2)
				q_KL_targ = self.softmin(q1_KL_targ, q2_KL_targ)
            
            if self.algo == 'cca':    
                backup_ref =  r  +  self.gamma * (1 - done) *  q_ref_targ 
                
                backup_KL =  (1 - done_KL) * TD_err + self.gamma * (1 - done) *  q_KL_targ
                                
                diff_1 = q_ref_1 - backup_ref
                diff_2 = q_ref_2 - backup_ref
                lik_1 = - 0.5 * (1-self.gamma) * diff_1**2 
                lik_2 = - 0.5 * (1-self.gamma) * diff_2**2 
                backup_1 =  lik_1 + (1 - done_KL)* (1 - self.gamma)/self.prec * TD_err + self.gamma * (1 - done) *  q_targ
                backup_2 =  lik_2 + (1 - done_KL)* (1 - self.gamma)/self.prec * TD_err + self.gamma * (1 - done) *  q_targ 
                                                
            else:
                backup =  r + self.gamma * (1 - done) * (q_targ - 1 / self.beta * logp_a_prim) # SAC update           

        # MSE loss against Bellman backup  
        if self.algo == 'cca':   
            loss_q1_ref = 0.5 * ((q_ref_1 - backup_ref)**2).mean() 
            loss_q2_ref = 0.5 * ((q_ref_2 - backup_ref)**2).mean() 
            loss_q1_KL = 0.5 * ((q_KL_1 - backup_KL)**2).mean() 
            loss_q2_KL = 0.5 * ((q_KL_2 - backup_KL)**2).mean() 
            loss_q1 = 0.5 * ((q_1 - backup_1)**2 ).mean() 
            loss_q2 = 0.5 * ((q_2 - backup_2)**2 ).mean()
            loss_q = loss_q1 + loss_q2 + loss_q1_ref + loss_q2_ref + loss_q1_KL +  loss_q2_KL  
            # Useful info for logging
			q_info = dict(Q1Vals=q_1.detach().numpy(),
						  Q2Vals=q_2.detach().numpy(),
						  Q1Vals_ref=q_ref_1.detach().numpy(),
						  Q2Vals_ref=q_ref_2.detach().numpy(),
						  Q1Vals_KL=q_KL_1.detach().numpy(),
						  Q2Vals_KL=q_KL_2.detach().numpy())
        else:
            loss_q1 = 0.5 * ((q_1 - backup)**2).mean() 
            loss_q2 = 0.5 * ((q_2 - backup)**2).mean() 
            loss_q = loss_q1 + loss_q2
            # Useful info for logging
			q_info = dict(Q1Vals=q_1.detach().numpy(),
						  Q2Vals=q_2.detach().numpy())

        
        
        return loss_q, q_info, log_p_info
    
    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        s = data['obs']
        a, logp_a = self.actor.pi(s)
        if self.algo == 'cca':   
			q1_pi = self.actor.q_ref_1(s, a)
			q2_pi = self.actor.q_ref_2(s, a)
			q_pi = self.softmin(q1_pi, q2_pi)
			ELBO_1 = self.actor.q_1(s, a)
			ELBO_2 = self.actor.q_2(s, a)
			ELBO = self.softmin(ELBO_1 , ELBO_2)
			
			a_test, logp_a_test = self.actor.pi_test(s)
			q1_pi_test = self.actor.q_ref_1(s, a_test)
			q2_pi_test = self.actor.q_ref_2(s, a_test)
			q_pi_test = self.softmin(q1_pi_test, q2_pi_test)
			
			# Entropy-regularized policy loss
			loss_pi = (logp_a - self.beta * (q_pi + ELBO)).mean() +  (logp_a_test - self.beta * q_pi_test).mean()
		else:
			q1_pi = self.actor.q_1(s, a)
			q2_pi = self.actor.q_2(s, a)
			q_pi = self.softmin(q1_pi, q2_pi)
			loss_pi = (logp_a - self.beta * q_pi).mean()
        
        # Useful info for logging
        pi_info = dict(LogPi=logp_a.detach().numpy())
        
        return loss_pi, pi_info

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
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.q_params:
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
            
    def get_action(self, obs, deterministic=False, test=False):
        return self.actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic, test)     

    def test_agent(self):
        for j in range(self.num_test_episodes):
            obs, done, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(done or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                obs, r, done, _ = self.test_env.step(self.get_action(obs, deterministic=True, test=False))
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
            # use the learned policy. 
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
            if False: #self.t > self.update_after:
                log_p_obs = self.state_probs.score_samples([s_prim])[0]
            else:
                log_p_obs = 0
            self.replay_buffer.store(s, a, r, log_p_obs, s_prim, done, ep_len)

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
                    if self.algo == 'cca':   
						state_batch_size = min(self.replay_buffer.size, 1000)   
						state_batch = self.replay_buffer.sample_batch(state_batch_size)['obs2']
						self.state_probs = KernelDensity(kernel='gaussian',                                                   
														 bandwidth=self.bandwidth).fit(state_batch.detach().numpy())               
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            # End of epoch handling
            if (self.t+1) % self.steps_per_epoch == 0:
                epoch = (self.t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                if self.algo != 'cca':
					self.agent.pi_test = deepcopy(self.agent.pi)
                self.test_agent()

                # Log info about epoch
                if self.t >= self.update_after:
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('EpRet', with_min_and_max=True)
                    self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('TestEpLen', average_only=True)
                    self.logger.log_tabular('TotalEnvInteracts', self.t)
                    if self.algo == 'cca':   
						self.logger.log_tabular('LogPObs', with_min_and_max=True)
                    self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                    self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                    if self.algo == 'cca':   
						self.logger.log_tabular('Q1Vals_ref', with_min_and_max=True)
						self.logger.log_tabular('Q2Vals_ref', with_min_and_max=True)
						self.logger.log_tabular('Q1Vals_KL', with_min_and_max=True)
						self.logger.log_tabular('Q2Vals_KL', with_min_and_max=True)
                    self.logger.log_tabular('LogPi', average_only=True)   
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossQ', average_only=True)
                    self.logger.log_tabular('Time', time.time()-start_time)
                    self.logger.dump_tabular()
                    
            self.t += 1

def cca(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
        logger_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, beta=10, prec=0.3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        save_freq=1, bandwidth=0.1, algo='cca'):
            
    trainer = CCA_Trainer(env_fn=env_fn, 
                            actor_critic=actor_critic,
                       ac_kwargs=ac_kwargs, 
                       logger_kwargs=logger_kwargs,
                       seed = seed,
                       start_steps=start_steps, 
                       steps_per_epoch=steps_per_epoch, 
                       update_after=update_after, 
                       update_every=update_every,
                       epochs=epochs, 
                       max_ep_len=max_ep_len,
                       gamma=gamma,
                       beta=beta,
                       prec=prec,
                       replay_size = replay_size,
                       batch_size=batch_size,
                       bandwidth=bandwidth,
                       algo=algo)

    trainer.train()
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='cca') # 'cca', 'sac'
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)   
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--prec', type=float, default=1)
    parser.add_argument('--bandwidth', type=float, default=0.1) 
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--update_after', type=int, default=10000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='cca')
    args = parser.parse_args()
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    

    torch.set_num_threads(torch.get_num_threads())
    
    ac_kwargs = dict(hidden_sizes=[args.hid]*args.l, activation=torch.nn.ReLU) 
    
    env_fn = lambda : gym.make(args.env)
    
    cca( env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, **args)

