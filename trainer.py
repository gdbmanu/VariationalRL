import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma
from environment import Environment
from agent import Transition
from sklearn.neighbors import KernelDensity

import torch
import time

class KNN_prob():
    def __init__(self, data, k=10):
        self.data = data
        self.n, self.d = self.data.shape
        self.V = np.pi**(self.d/2) / gamma(self.d/2 + 1)
        self.k = k
        self.q = k/self.n        
        #print(self.n, self.d, self.q)
    def __call__(self, x):
        dists = np.sqrt(np.sum((x - self.data)**2,1))
        dist = np.quantile(dists, self.q)
        return self.k/ (self.n * self.V * dist**self.d)

class Trainer():

    def __init__(self, agent,
                 OBS_LEAK=1e-3,
                 EPSILON=1e-3,
                 N_PART=10,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi=False,
                 augmentation=True,
                 KNN_prob=False,
                 retain_present=False,
                 retain_trajectory=False,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 clip_gradients=False):
        self.agent = agent
        self.nb_trials = 0
        self.init_trial(update=False)
        if self.agent.isDiscrete:
            self.nb_visits = np.zeros(self.agent.env.N_obs)
            self.obs_score = np.zeros(self.agent.env.N_obs)
            self.nb_visits_final = np.zeros(self.agent.env.N_obs)
            self.obs_score_final = np.zeros(self.agent.env.N_obs)
        else:
            self.mem_obs = []
        self.mem_obs_final = []
        if self.agent.continuousAction:
            self.mem_act = []
            self.mu_act = np.zeros(self.agent.N_act)
            self.Sigma_act = np.diag(np.ones(self.agent.N_act))
        self.mem_total_reward = []
        self.mem_total_reward_test = []
        self.mem_mean_rtg = []
        self.mem_KL_final = []
        self.mem_t_final = []
        self.mem_t_final_test = []
        self.OBS_LEAK = OBS_LEAK
        self.EPSILON = EPSILON
        self.N_PART=N_PART
        self.HIST_HORIZON = agent.HIST_HORIZON
        self.mem_V = {}
        self.ref_prob = ref_prob
        self.final = final
        self.monte_carlo = monte_carlo
        self.Q_learning = Q_learning
        self.KL_reward = KL_reward
        self.ignore_pi = ignore_pi
        self.augmentation = augmentation
        self.KNN_prob = KNN_prob
        self.retain_present = retain_present
        self.retain_trajectory = retain_trajectory
        self.KL_correction = KL_correction
        self.Q_ref_correction = Q_ref_correction
        self.BATCH_SIZE=BATCH_SIZE
        self.clip_gradients = clip_gradients

    def init_trial(self, update=True):
        if update:
            self.nb_trials += 1
        self.total_reward = 0
        self.trajectory = []
        self.action_history = []
        self.actions_set_history = []
        self.reward_history = []
        self.rtg_history = []
        self.ktg_history = []

    def calc_state_probs(self, MAX_SAMPLE = 100000):
        # return self.nb_visits/np.sum(self.nb_visits)
        if self.agent.isDiscrete:
            return self.obs_score / np.sum(self.obs_score)
        else:
            b_inf = min(MAX_SAMPLE, len(self.agent.memory))
            #if b_inf <= MAX_SAMPLE:
            #    return KNN_prob(np.array(self.mem_obs[-b_inf:]), k=10)
            #else:
            if b_inf>0:
                obs_batch, _, _, _ = self.memory_sample(b_inf)
            else:
                obs_batch = self.mem_obs
            if self.nb_trials>10 and self.KNN_prob:    
                return KNN_prob(obs_batch, k=10)
            #else:
            #    print('OK')
            #    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(obs_batch)
            #    f = kde.score_samples
            #    return lambda x:np.exp(f([x]))
            else:
                #b_inf = min(self.HIST_HORIZON, len(self.mem_obs))
                #mu = np.mean(self.mem_obs[-b_inf:], axis = 0)
                mu = np.mean(obs_batch, axis = 0)
                #print('mu', mu)
                #eps = 1e-5
                #Sigma = (1-eps) * np.cov(np.array(self.mem_obs[-b_inf:]).T) + eps * np.diag(np.ones(self.agent.N_obs))
                #print('Sigma', Sigma)
                #try:
                #    rv = multivariate_normal(mu, Sigma)
                #except:
                if self.agent.env.observation_space.shape[0] < 5:
                    try:
                        var = np.var(obs_batch[-b_inf:], axis=0)
                        rv = multivariate_normal(mu, var)
                    except:
                        rv = multivariate_normal(mu, np.ones(len(mu)))
                else:
                    rv = multivariate_normal(mu, np.ones(len(mu)))
                return rv.pdf  
        
        
    def calc_final_state_probs(self):
        # return self.nb_visits/np.sum(self.nb_visits)
        if self.agent.isDiscrete:
            return self.obs_score_final / np.sum(self.obs_score_final)
        else:                
            b_inf = min(int(1/self.OBS_LEAK), len(self.mem_obs_final))
            if self.KNN_prob and self.nb_trials>10:
                return KNN_prob(np.array(self.mem_obs_final[-b_inf:]), k=10)
            else:
                mu = np.mean(self.mem_obs_final[-b_inf:], axis=0)
                eps = 1e-5
                #Sigma = np.cov(np.array(self.mem_obs_final).T)
                #Sigma = (1-eps) * np.cov(np.array(self.mem_obs_final[-b_inf:]).T) + eps * np.diag(np.ones(self.agent.N_obs))
                #try:
                #    rv = multivariate_normal(mu, Sigma)
                #except:
                try:
                    var = np.var(self.mem_obs_final[-b_inf:], axis=0)
                    rv = multivariate_normal(mu, var)
                except:
                    rv = multivariate_normal(mu, np.ones(len(mu)))          
                return rv.pdf # !! TODO a verifier 

        
    def calc_ref_probs(self, obs, EPSILON=1e-3, final=False):
        if self.ref_prob == 'unif':
            # EXPLORATION DRIVE
            if self.agent.isDiscrete:
                if self.augmentation:
                    p = np.zeros(self.agent.N_obs)
                    if self.final:
                        ## !!!! A revoir TODO !!!!!!!!
                        #p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0)
                        p[np.where(self.nb_visits_final > 0)] = 1 / np.sum(self.nb_visits_final > 0)
                    else:
                        p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0)
                else:
                    p = np.ones(self.agent.N_obs) / self.agent.N_obs
                return p
            else:
                if self.augmentation:
                    b_inf = len(self.mem_obs) #min(self.HIST_HORIZON, len(self.mem_obs))
                    high = np.max(self.mem_obs[-b_inf:], axis = 0)
                    low = np.min(self.mem_obs[-b_inf:], axis = 0)
                    if np.prod(high - low) > 0:                   
                        return 1 / np.prod(high - low)
                    else:
                        return 1
                else:
                    return 1 / np.prod(self.agent.env.observation_space.high - \
                                       self.agent.env.observation_space.low)               
        else:
            # SET POINT
            if self.agent.isDiscrete:
                ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
                ref_probs[int(self.ref_prob)] += (1 - EPSILON)
                return ref_probs
            else:
                return None # TODO

    def calc_actions_prob(self):
        b_inf = min(self.HIST_HORIZON, len(self.mem_act))
        if b_inf > 0:
            mu = np.mean(self.mem_act[-b_inf:], axis = 0)
            if b_inf >= self.HIST_HORIZON:
                Sigma = np.cov(np.array(self.mem_act[-b_inf:]).T)
            else:
                Sigma = np.diag(np.ones(len(mu)))
        else:
            mu = np.zeros(self.agent.N_act)
            Sigma = np.diag(np.ones(len(mu)))
        return mu, Sigma

    def set_actions_set(self):
        #if False:
        #    
        #else:
        #    mu = np.zeros(self.agent.N_act)
        #    Sigma =  np.diag(np.ones(self.agent.N_act)) ## * 0.3**2 ## !!
        actions_set = []
        for indice_act in range(0): #3 * self.N_PART // 4):
            if len(self.mu_act) == 1:
                act = self.mu_act + np.random.normal() * np.sqrt(self.Sigma_act) #np.random.normal(mu, Sigma)
                act = np.array(act[0])
            else:
                act = np.random.multivariate_normal(self.mu_act, self.Sigma_act)
            act = np.clip(act, self.agent.act_low, self.agent.act_high)
            actions_set.append(act)
        for indice_act in range(self.N_PART): # // 4): #self.N_PART // 2):
            act = self.agent.env.action_space.sample()
            actions_set.append(act)
        return actions_set
    
    def KL(self, obs, done=False, verbose=False):
        if self.final: # Only final state for probability calculation
            if done:
                final_state_probs = self.calc_final_state_probs()
                ref_probs = self.calc_ref_probs(obs, EPSILON=self.EPSILON)
                if self.agent.isDiscrete:
                    return np.log(final_state_probs[obs]) - np.log(ref_probs[obs])
                else:
                    if verbose:
                        print('obs :', obs, ', KL loss : ', np.log(final_state_probs(obs)) - np.log(ref_probs))
                    return np.log(final_state_probs(obs)) - np.log(ref_probs)
            else:
                return 0  # self.agent.Q_KL_tab[past_obs, a]
        else:
            if self.agent.isDiscrete:
                state_probs = self.calc_state_probs()
                ref_probs = self.calc_ref_probs(obs, EPSILON=self.EPSILON)
                return np.log(state_probs[obs]) - np.log(ref_probs[obs])  # +1
            else:
                state_probs = self.state_probs
                ref_probs = self.ref_probs
                try:
                    KL_out = max(np.log(state_probs(obs)) - np.log(ref_probs), -100)
                except:
                    KL_out = -100
                return KL_out

    def calc_sum_future_KL(self, obs, obs_or_time, done, tf=False, actions_set=None):
        sum_future_KL = self.KL(obs, done=done)
        if tf:
            sum_future_KL = torch.FloatTensor([sum_future_KL])      
        if not done:
            if self.agent.continuousAction and actions_set is None:
                actions_set = self.set_actions_set()
            next_values = self.agent.set_Q_obs(obs_or_time, Q=self.agent.Q_KL, tf=tf, actions_set=actions_set)
            next_sum = self.agent.softmax_expectation(obs_or_time, next_values, tf=tf, actions_set=actions_set)
            sum_future_KL += self.agent.GAMMA * next_sum
        return sum_future_KL
    
    def calc_KL_logPolicy_loss_tf(self, past_obs, past_action, past_actions_set, obs, 
                                  future_actions_set, done = False, sum_future_KL = None):
        if sum_future_KL is None:
            sum_future_KL = self.calc_sum_future_KL(obs, obs, done, tf=False, actions_set=future_actions_set)             
        #print('sum_future_KL', sum_future_KL)
        sum_future_KL = torch.FloatTensor([sum_future_KL])      
        logPolicy = self.agent.logSoftmax(past_obs, past_action, actions_set=past_actions_set, tf = True)
        return logPolicy * sum_future_KL

    def online_KL_err(self, past_obs_or_time, past_action, obs, obs_or_time, done=False):
        sum_future_KL = self.calc_sum_future_KL(obs, obs_or_time, done)
        return self.agent.Q_KL(past_obs_or_time, past_action) - sum_future_KL

    def KL_loss_tf(self, KL_pred_tf, obs, done, actions_set=None): # Q TD_error
        sum_future_KL = self.calc_sum_future_KL(obs, obs, done, tf=False, actions_set=actions_set) # not tf=True!!!
        sum_future_KL_tf = torch.FloatTensor([sum_future_KL])
        return torch.pow(KL_pred_tf - sum_future_KL_tf, 2)
            
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time=None):
        return 0

    def calc_sum_future_rewards(self, reward, obs_or_time, done, tf=False, actions_set=None):
        sum_future_rewards = reward
        if tf:
            sum_future_rewards = torch.FloatTensor([sum_future_rewards])
        if not done:
            if self.agent.continuousAction and actions_set is None:
                actions_set = self.set_actions_set()
            next_values = self.agent.set_Q_obs(obs_or_time,
                                               Q=self.agent.Q_ref,
                                               tf=tf, 
                                               actions_set=actions_set)
            sum_future_rewards += self.agent.GAMMA * self.agent.softmax_expectation(obs_or_time, 
                                                                                    next_values, 
                                                                                    tf=tf, 
                                                                                    actions_set=actions_set)
        
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs_or_time, past_action):
        return self.agent.PREC * (self.agent.Q_ref(past_obs_or_time, past_action) - sum_future_rewards)

    def online_TD_err_ref(self, past_obs_or_time, past_action, obs_or_time, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs_or_time, done)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)

    def Q_ref_loss_tf(self, Q_ref_pred_tf, obs, reward, done, actions_set=None):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done, actions_set=actions_set)
        sum_future_rewards_tf = torch.FloatTensor([sum_future_rewards])
        return 0.5 * torch.pow(Q_ref_pred_tf - sum_future_rewards_tf, 2)

    def calc_TD_err_var(self, sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        return self.agent.PREC * (self.agent.Q_var(past_obs_or_time, past_action) - sum_future_rewards) \
                                  + 1 / self.agent.BETA * mult_Q * sum_future_KL

    def online_TD_err_var(self, past_obs, past_obs_or_time, past_action, obs, obs_or_time, reward, done=False):      
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs_or_time, done)
        sum_future_KL = self.agent.Q_KL(past_obs_or_time, past_action)
        return self.calc_TD_err_var(sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action)

    def Q_var_loss_tf(self, Q_var_pred_tf, KL_pred_tf, obs, reward, done):

        #if self.final:
        #    if self.agent.GAMMA == 1:
        #        LAMBDA = 10 #self.agent.env.total_steps / 2
        #    else:
        #        LAMBDA = 1 / self.agent.GAMMA
        #else:
        #    LAMBDA = 1

        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done, tf=False)
        sum_future_rewards_tf = torch.FloatTensor([sum_future_rewards])

        if self.Q_learning:
            return 0.5 * self.agent.PREC * torch.pow(Q_var_pred_tf - sum_future_rewards_tf, 2)
        else:
            #return 0.5 * self.agent.PREC * torch.pow(sum_future_rewards_tf - Q_var_pred_tf, 2) + 1 / self.agent.BETA * KL_pred_tf
            return 0.5 * self.agent.PREC * torch.pow(Q_var_pred_tf - sum_future_rewards_tf + 1 / self.agent.BETA / self.agent.PREC * KL_pred_tf, 2)

    def memory_sample(self, batch_size):
        transitions = self.agent.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        obs_batch = np.array(batch.obs) #torch.cat(batch.obs)                            
        act_batch = np.array(batch.action) #torch.cat(batch.action)                            
        sum_future_KL_batch = torch.cat(batch.sum_future_KL)                            
        #R_tilde_batch = torch.cat(batch.R_tilde)
        if self.agent.do_reward:                            
            sum_future_rewards_batch = torch.cat(batch.sum_future_rewards)
        else:
            sum_future_rewards_batch = torch.zeros((batch_size, 1))

        
        return obs_batch, act_batch, sum_future_KL_batch, sum_future_rewards_batch


    def online_update(self, past_obs, past_action, obs, reward, done, past_time, current_time, actions_set=None):
        if self.agent.isTime:
            past_obs_or_time = past_time
            obs_or_time = current_time
        else:
            past_obs_or_time = past_obs
            obs_or_time = obs
        if self.agent.isDiscrete:
            if not self.Q_learning:
                 # 30 #
                self.agent.Q_KL_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * self.online_KL_err(past_obs_or_time,
                                                                                              past_action,
                                                                                              obs,
                                                                                              obs_or_time,
                                                                                              done=done)
            self.agent.Q_ref_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * self.online_TD_err_ref(past_obs_or_time,
                                                                                                  past_action,
                                                                                                  obs_or_time,
                                                                                                  reward,
                                                                                                  done=done)
            self.agent.Q_var_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * self.agent.Q_VAR_MULT * self.online_TD_err_var(past_obs,
                                                                                                  past_obs_or_time,
                                                                                                  past_action,
                                                                                                  obs,
                                                                                                  obs_or_time,
                                                                                                  reward,
                                                                                                  done=done)
        else:
            if actions_set is not None:
                future_actions_set = self.set_actions_set()
            else:
                future_actions_set = None
                
            if not self.Q_learning:
                if self.agent.get_time() == 1:
                    tic = time.clock()
                KL_pred_tf = self.agent.Q_KL(past_obs, past_action, tf=True)
                loss_KL = self.KL_loss_tf(KL_pred_tf, obs, done, actions_set=future_actions_set)
                loss_KL.backward()
                for param in self.agent.Q_KL_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.agent.Q_KL_optimizer.step()
                if self.agent.get_time() == 1:
                    toc = time.clock()
                    print('KL step', toc - tic)

            if self.agent.do_reward:
                if self.agent.get_time() == 1:
                    tic = time.clock()
                Q_ref_pred_tf = self.agent.Q_ref(past_obs, past_action, tf=True)
                loss_Q_ref = self.Q_ref_loss_tf(Q_ref_pred_tf, obs, reward, done, actions_set=future_actions_set)
                loss_Q_ref.backward()
                for param in self.agent.Q_ref_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.agent.Q_ref_optimizer.step()
                if self.agent.get_time() == 1:
                    toc = time.clock()
                    print('Q_ref step', toc - tic)

            if not self.Q_learning:
                if self.agent.get_time() == 1:
                    tic = time.clock()
                Q_var_pred_tf = self.agent.Q_var(past_obs, past_action, tf=True)
                KL_pred_tf = self.calc_sum_future_KL(past_obs, past_obs, done, tf=True, actions_set=actions_set)
                KL_pred_tf -= torch.FloatTensor([self.KL(past_obs, done=done)])
                loss_Q_var = self.Q_var_loss_tf(Q_var_pred_tf, KL_pred_tf, obs, reward, done)
                loss_Q_var.backward()
                for param in self.agent.Q_var_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.agent.Q_var_optimizer.step()
                if self.agent.get_time() == 1:
                    toc = time.clock()
                    print('Q_var step', toc - tic)
            else:
                self.agent.Q_var = self.agent.Q_ref
                
            self.agent.Q_KL_optimizer.zero_grad()
            self.agent.Q_ref_optimizer.zero_grad()
            self.agent.Q_var_optimizer.zero_grad()

    def monte_carlo_update(self, done):
        if done:
            final_time = self.agent.get_time()
            liste_KL = np.zeros(final_time+1)
            liste_sum_KL = np.zeros(final_time)
            liste_rtg = np.zeros(final_time+1)
            
            if not self.Q_learning:
                # FIRST LOOP
                for time in range(final_time):
                    new_obs = self.trajectory[time + 1]
                    test_done = final_time == time + 1
                    liste_KL[time] = self.KL(new_obs, done=test_done)
                if self.KL_correction:
                    liste_KL[final_time] = self.calc_sum_future_KL(new_obs, new_obs, False)
                # SECOND LOOP
                for time in range(final_time):
                    if False: #self.final:
                        liste_sum_KL[time] = np.sum(np.array(liste_KL[time:]))
                    else:
                        liste_sum_KL[time]  = np.sum(np.array(liste_KL[time:]) * \
                                           self.agent.GAMMA **(np.arange(time, final_time+1) - time))
                    self.ktg_history.append(liste_sum_KL[time])
                    #Q_var_pred = self.agent.Q_var(past_obs_or_time, past_action)
                    #if self.nb_trials > 20:
                    #    R_tilde = Q_var_pred - 1/ self.agent.BETA * sum_future_KL
                    #else:
                    #    R_tilde = - 1/ self.agent.BETA * sum_future_KL    
                    
            if self.agent.do_reward:
                ## !!TEST!!
                if self.Q_ref_correction :
                    if final_time == 1600:
                        print('OK')
                        self.reward_history[final_time-1]=self.calc_sum_future_rewards(self.reward_history[final_time-1], 
                                                                                     new_obs, 
                                                                                     False)
                    
                for time in range(final_time):
                    liste_rtg[time] = np.sum(np.array(self.reward_history[time:]) * \
                                          self.agent.GAMMA **(np.arange(time, final_time) - time))
                    self.rtg_history.append(liste_rtg[time])
                    
            # THIRD LOOP
            for time in range(final_time):                ## !!!! faux dans le cas "full KL" et "full reward" !!!! TODO ##
                past_obs = self.trajectory[time]
                if self.agent.isTime:
                    past_obs_or_time = time
                else:
                    past_obs_or_time = self.trajectory[time]
                past_action = self.action_history[time]
                #actions_set = self.actions_set_history[time]
                sum_future_KL = liste_sum_KL[time]
                sum_future_rewards = liste_rtg[time]

                if not self.agent.isDiscrete:
                    self.agent.memory.push(past_obs, 
                                           past_action, 
                                           torch.FloatTensor([sum_future_KL]), 
                                           torch.FloatTensor([sum_future_rewards]),
                                           0
                                          )


                if self.nb_trials > 10:
                    if self.agent.isDiscrete:
                        TD_err_ref = self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)
                        self.agent.Q_ref_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * TD_err_ref

                        TD_err_var = self.calc_TD_err_var(sum_future_rewards,
                                                          sum_future_KL, 
                                                          past_obs,
                                                          past_obs_or_time,
                                                          past_action)
                        self.agent.Q_var_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * TD_err_var
                    else:                        
                        if len(self.agent.memory) > self.BATCH_SIZE:
                            obs_batch, act_batch, sum_future_KL_batch, sum_future_rewards_batch = self.memory_sample(self.BATCH_SIZE)                            
                            
                            if self.retain_present :
                                obs_batch[0] = past_obs
                                act_batch[0] = past_action
                                if not self.Q_learning:
                                    sum_future_KL_batch[0] = torch.FloatTensor([sum_future_KL])
                                if self.agent.do_reward:  
                                    sum_future_rewards_batch[0] = torch.FloatTensor([sum_future_rewards])
                                    
                            if self.retain_trajectory:
                                num_sample = np.random.randint(final_time)
                                obs_batch[1] = self.trajectory[num_sample]
                                act_batch[1] = self.action_history[num_sample]
                                if not self.Q_learning:
                                    sum_future_KL_batch[1] = torch.FloatTensor([liste_sum_KL[num_sample]])
                                if self.agent.do_reward:  
                                    sum_future_rewards_batch[1] = torch.FloatTensor([liste_rtg[num_sample]])                            
                            
                            Q_KL_pred_tf = self.agent.Q_KL(obs_batch,
                                                         act_batch,
                                                         tf=True)
                            loss_Q_KL = torch.sum(0.5 *  torch.pow(Q_KL_pred_tf - sum_future_KL_batch, 2))
                            self.agent.Q_KL_optimizer.zero_grad()
                            loss_Q_KL.backward()
                            if self.clip_gradients:
                                for param in self.agent.Q_KL_nn.parameters():
                                    param.grad.data.clamp_(-1, 1)
                            self.agent.Q_KL_optimizer.step()
                            
                            Q_ref_pred_tf = self.agent.Q_ref(obs_batch,
                                                         act_batch,
                                                         tf=True)
                            loss_Q_ref = torch.sum(0.5 * self.agent.PREC * torch.pow(Q_ref_pred_tf - sum_future_rewards_batch, 2))
                            self.agent.Q_ref_optimizer.zero_grad()
                            loss_Q_ref.backward()
                            if self.clip_gradients:
                                for param in self.agent.Q_ref_nn.parameters():
                                    param.grad.data.clamp_(-1, 1)
                            self.agent.Q_ref_optimizer.step()
                            
                            if self.Q_learning:    
                                self.agent.Q_var = self.agent.Q_ref
                            else:
                                Q_var_pred_tf = self.agent.Q_var(obs_batch,
                                                             act_batch,
                                                             tf=True)
                                #print(Q_var_pred_tf.detach().numpy().shape)
                                R_tilde_batch = Q_var_pred_tf.detach() - 1/ self.agent.BETA * sum_future_KL_batch
                                loss_Q_var = torch.sum(
                                      0.5 * self.agent.PREC *  torch.pow(Q_var_pred_tf - sum_future_rewards_batch, 2) 
                                      + 0.5 * torch.pow(Q_var_pred_tf - R_tilde_batch, 2)
                                                      )
                                self.agent.Q_var_optimizer.zero_grad()
                                loss_Q_var.backward()
                                # TODO : à tester
                                if self.clip_gradients:
                                    for param in self.agent.Q_var_nn.parameters():
                                        param.grad.data.clamp_(-1, 1)
                                self.agent.Q_var_optimizer.step() 
                    #print('')
                    #print(time) 
                    #print('sum KL', sum_future_KL)
                    #print('future rewards', sum_future_rewards_batch.detach().numpy()[0])
                    #print('R_tilde', R_tilde_batch.detach().numpy())
                    #print('Q_var', Q_var_pred_tf.detach().numpy()[0])
                    #print('Loss', loss_Q_var.detach().numpy())


                if False: #not self.agent.isDiscrete:
                    BATCH_SIZE = 20 # final_time
                    # THIRD LOOP
                    for initial_batch_time in range(0, final_time, BATCH_SIZE):
                        #print('batch_time', initial_batch_time)
                        final_batch_time = min(initial_batch_time + BATCH_SIZE, final_time)
                        current_batch_size = final_batch_time - initial_batch_time
                        if not self.Q_learning: # !!!!TODO : tests!!!!
                            #KL_pred_tf = self.agent.Q_KL(self.trajectory[initial_batch_time:final_batch_time],
                            #                       self.action_history[initial_batch_time:final_batch_time],
                            #                       tf=True)
                            sum_future_KL_slice = sum_future_KL_list[initial_batch_time:final_batch_time]
                            sum_future_KL_tf = torch.FloatTensor([sum_future_KL_slice]).view((current_batch_size, 1))
                            '''if False:
                                loss_KL_list = 0.5 * torch.pow(KL_pred_tf - sum_future_KL_tf, 2)
                                loss_KL = torch.sum(loss_KL_list)
                                self.agent.Q_KL_optimizer.zero_grad()
                                loss_KL.backward()
                                self.agent.Q_KL_optimizer.step()'''

                        if self.agent.do_reward:
                            
                            sum_future_rewards_slice = sum_future_rewards_list[initial_batch_time:final_batch_time]
                            sum_future_rewards_tf = torch.FloatTensor([sum_future_rewards_slice]).view((current_batch_size,
                                                                                                        1))
                            if self.Q_learning:
                                Q_ref_pred_tf = self.agent.Q_ref(self.trajectory[initial_batch_time:final_batch_time],
                                                             self.action_history[initial_batch_time:final_batch_time],
                                                             tf=True)#.view((final_time, 1))
                                loss_Q_ref = torch.sum(0.5 * torch.pow(Q_ref_pred_tf - sum_future_rewards_tf, 2))
                                #loss_Q_ref = torch.nn.MSELoss()(Q_ref_pred_tf, sum_future_rewards_tf)
                                if initial_batch_time == 0:
                                    print('Q_ref_pred_tf ', Q_ref_pred_tf[0])
                                    print('sum_future_rewards_tf ', sum_future_rewards_tf[0])
                                    print('loss_Q_ref', loss_Q_ref)
                                self.agent.Q_ref_optimizer.zero_grad()
                                loss_Q_ref.backward()
                                self.agent.Q_ref_optimizer.step()
                        else:
                            sum_future_rewards_tf = torch.zeros((current_batch_size, 1))

                        if self.Q_learning:
                            self.agent.Q_var = self.agent.Q_ref
                        else:
                            #if False:
                            #    for time in range(initial_batch_time, final_batch_time):
                            #        past_obs = self.trajectory[time]
                            #        obs = self.trajectory[time + 1]
                            #        past_action = self.action_history[time]
                            #        past_actions_set = self.actions_set_history[time]
                            #        test_done = final_time == time + 1
                            #        if test_done:
                            #            future_actions_set = None
                            #        else:
                            #            future_actions_set = self.actions_set_history[time + 1]
                            #        '''loss_KL_tf = self.calc_KL_logPolicy_loss_tf(past_obs,
                            #                                                    past_action,
                            #                                                    past_actions_set,
                            #                                                    obs,
                            #                                                    future_actions_set,
                            #                                                    done = test_done,
                            #                                                    sum_future_KL = sum_future_KL_list[time]
                            #                                                   )'''
                            #        '''loss_KL_tf = self.calc_sum_future_KL(past_obs, past_obs, 
                            #                                             test_done, tf=True, 
                            #                                             actions_set=actions_set) 
                            #        if time > 0:
                            #            loss_KL_tf -= torch.FloatTensor([liste_KL[time-1]])'''
                            #        loss_KL_tf = self.calc_sum_future_KL(obs, obs,
                            #                                             test_done, tf=True,
                            #                                             actions_set=future_actions_set)
                            #        if time == initial_batch_time:
                            #            loss_KL_tf_list = loss_KL_tf
                            #        else:
                            #            loss_KL_tf_list = torch.cat((loss_KL_tf_list, loss_KL_tf))
                            
                            # from DQN tuto
                            transitions = self.agent.memory.sample(BATCH_SIZE)
                            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                            # detailed explanation). This converts batch-array of Transitions
                            # to Transition of batch-arrays.
                            batch = Transition(*zip(*transitions))
                            #obs_batch = self.trajectory[initial_batch_time:final_batch_time]
                            obs_batch = batch.obs #torch.cat(batch.obs)
                            #act_batch = self.action_history[initial_batch_time:final_batch_time]
                            act_batch = batch.action #torch.cat(batch.action)
                            #sum_future_KL_batch = sum_future_KL_tf.view((current_batch_size, 1))
                            sum_future_KL_batch = torch.cat(batch.sum_future_KL)
                            
                            if self.agent.do_reward:
                                #sum_future_rewards_batch = sum_future_rewards_tf
                                sum_future_rewards_batch = torch.cat(batch.sum_future_rewards)
                            else:
                                sum_future_rewards_batch = torch.zeros((current_batch_size, 1))

                            Q_var_pred_tf = self.agent.Q_var(obs_batch,
                                                         act_batch,
                                                         tf=True)

                            #if False: #!! TODO sans TD-err
                            #    if self.agent.do_reward:
                            #        sum_future_rewards_tf = Q_ref_pred_tf.detach()
                            #    else:
                            #        sum_future_rewards_tf = torch.zeros((current_batch_size, 1))

                            # 1 #
                            #loss_Q_var = torch.sum(self.agent.PREC * 0.5 * torch.pow(Q_var_pred_tf - sum_future_rewards_tf, 2) \
                            #                      + 1 / self.agent.BETA * loss_KL_tf_list.view((current_batch_size, 1)))
                            
                            # 2 #
                            #loss_Q_var = torch.sum(
                            #    0.5 * self.agent.PREC * torch.pow(Q_var_pred_tf - sum_future_rewards_tf + \
                            #                    + 1 / self.agent.BETA / self.agent.PREC * loss_KL_tf_list.view((current_batch_size, 1)), 2) \
                            #    )
                            
                            # 3 #
                            R_tilde_tf = Q_var_pred_tf.detach() - 1/ self.agent.BETA * sum_future_KL_batch
                            loss_Q_var = torch.sum(
                                  0.5 * self.agent.PREC *  torch.pow(Q_var_pred_tf - sum_future_rewards_batch, 2) 
                                + 0.5 * torch.pow(Q_var_pred_tf - R_tilde_tf, 2)
                                                  )
                            self.agent.Q_var_optimizer.zero_grad()
                            loss_Q_var.backward()
                            self.agent.Q_var_optimizer.step()



    def run_episode(self, train=True, render=False, verbose=False):
        self.agent.init_env()
        self.init_trial(update=train)
        obs = self.agent.get_observation()
        self.trajectory.append(obs)
        tic = time.clock()

        while True:
            
            past_time = self.agent.get_time()
            past_obs = obs #self.agent.get_observation()
            if self.agent.continuousAction:
                actions_set = self.set_actions_set()
            else:
                actions_set = None

            ########### STEP #############
            if train:
                past_obs_or_time, past_action, obs_or_time, reward, done = self.agent.step(actions_set = actions_set)
            else:
                past_obs_or_time, past_action, obs_or_time, reward, done = self.agent.step(actions_set = actions_set, test=True)
            ##############################

            current_time = self.agent.get_time()
            obs = self.agent.get_observation()

            self.action_history.append(past_action)
            self.actions_set_history.append(actions_set)
            self.trajectory.append(obs)
            
            if self.agent.isDiscrete:
                self.nb_visits[obs] += 1
                self.obs_score *= 1 - self.OBS_LEAK
                self.obs_score[obs] += 1
                self.agent.num_episode += 1
            else:
                self.mem_obs += [obs]
                if len(self.mem_obs) > self.HIST_HORIZON:
                    del self.mem_obs[0]
                if self.agent.continuousAction:
                    self.mem_act += [past_action]
                    if len(self.mem_act) > self.HIST_HORIZON:
                        del self.mem_act[0]

            if done:
                if self.agent.isDiscrete:
                    self.nb_visits_final[obs] += 1
                    self.obs_score_final *= 1 - self.OBS_LEAK
                    self.obs_score_final[obs] += 1
                self.mem_obs_final += [obs]

            if past_time == 0 and train:
                self.state_probs = self.calc_state_probs()
                self.ref_probs = self.calc_ref_probs(obs)
                if self.agent.continuousAction and not self.KNN_prob:
                    self.mu_act, self.Sigma_act = self.calc_actions_prob()

            mem_reward = reward
            if not self.agent.do_reward:
                reward = 0
            if self.KL_reward:
                reward -= self.KL(obs, done=done)
            self.reward_history.append(reward)
            self.total_reward += mem_reward

            ########### LEARNING STEP #############
            if train:
                if self.monte_carlo:
                    self.monte_carlo_update(done)
                else:
                    self.online_update(past_obs, past_action, obs, reward, done, past_time, current_time, actions_set=actions_set)
            #######################################

            if render and type(self.agent.env) is not Environment:
                self.agent.env.render()

            if done:
                if train:
                    KL_final = self.KL(obs, done=done)
                    self.mem_KL_final.append(KL_final)
                    self.mem_t_final.append(current_time)
                    self.mem_total_reward.append(self.total_reward)
                    self.mem_mean_rtg.append(np.mean(self.rtg_history))
                    if verbose:
                        print('obs:', obs, 'final KL loss:', KL_final) #, 'final time:', current_time, 'total reward:', self.total_reward)     
                    if self.nb_trials % 100 == 0 and self.agent.isDiscrete and not self.agent.isTime:
                        V = np.zeros(self.agent.N_obs)
                        for obs in range(self.agent.N_obs):
                            V[obs] = self.agent.softmax_expectation(obs, self.agent.set_Q_obs(obs), actions_set=actions_set)
                        self.mem_V[self.nb_trials] = V
                else:
                    self.mem_t_final_test.append(current_time)
                    self.mem_total_reward_test.append(self.total_reward)
                break
        toc = time.clock()
        if verbose:
            print('Time elapsed :', toc-tic)


class Q_learning_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 N_PART=10,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=False,
                 augmentation=False,
                 KNN_prob=False,
                 retain_present=False,
                 retain_trajectory=False,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 clip_gradients=False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         N_PART=N_PART,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KNN_prob=KNN_prob,
                         retain_present=retain_present,
                         retain_trajectory=retain_trajectory,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         clip_gradients=clip_gradients)


class KL_Q_learning_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 N_PART=10,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=True,
                 augmentation=True,
                 KNN_prob=False,
                 retain_present=False,
                 retain_trajectory=False,
                 KL_correction=False,
                 Q_ref_correction=False,                 
                 BATCH_SIZE=20,
                 clip_gradients=False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         N_PART=N_PART,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KNN_prob=KNN_prob,
                         retain_present=retain_present,
                         retain_trajectory=retain_trajectory,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         clip_gradients=clip_gradients)


class One_step_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 N_PART=10,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi = False,
                 augmentation=True,
                 KNN_prob=False,
                 retain_present=False,
                 retain_trajectory=False,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 clip_gradients=False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         N_PART=N_PART,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=False,
                         KL_reward=KL_reward,
                         ignore_pi=ignore_pi,
                         augmentation=augmentation,
                         KNN_prob=KNN_prob,
                         retain_present=retain_present,
                         retain_trajectory=retain_trajectory,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         clip_gradients=clip_gradients)

    # agent.Q_var update # DEPRECATED ??
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time=None):
        pi = self.agent.softmax(past_obs)[a]
        state_probs = self.agent.calc_state_probs(past_obs)
        ref_probs = self.calc_ref_probs(past_obs, EPSILON=self.EPSILON)
        return (1 - pi) * self.agent.BETA * (np.log(state_probs[new_obs]) + pi / state_probs[new_obs] * 1 - np.log(ref_probs[new_obs]))


class Final_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 N_PART=10,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi = False,
                 augmentation=True,
                 KNN_prob=False,
                 retain_present=False,
                 retain_trajectory=False,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 clip_gradients=False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         N_PART=N_PART,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=Q_learning,
                         KL_reward=KL_reward,
                         ignore_pi=ignore_pi,
                         augmentation=augmentation,
                         KNN_prob=KNN_prob,
                         retain_present=retain_present,
                         retain_trajectory=retain_trajectory,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         clip_gradients=clip_gradients)

    # agent.Q_var update
    def KL_diff(self, past_obs, a, final_obs, done=False, past_time=None):
        if past_time is None:
            pi = self.agent.softmax(past_obs)[a]
        else:
            pi = self.agent.softmax(past_time)[a]
        if self.ignore_pi:
            return self.agent.BETA * self.agent.Q_KL[past_obs, a]
        else:
            return (1 - pi) * self.agent.BETA * self.agent.Q_KL[past_obs, a]  # self.KL(past_obs, a, final_obs, done)

    ## DEPRECATED?
    def BETA_err(self, past_obs, past_action, past_time=None):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        if self.agent.isTime:
            past_obs_or_time = past_time
        else:
            past_obs_or_time = past_obs
        return - mult_Q * (self.agent.Q_var(past_obs_or_time, past_action)
                           - self.agent.softmax_expectation(past_obs_or_time)) \
               * self.agent.Q_KL(past_obs, past_action)
