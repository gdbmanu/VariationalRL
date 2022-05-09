import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma
from environment import Environment
from sklearn.neighbors import KernelDensity

import torch
import time


class OnlineTrainer():

    def __init__(self, agent,
                 OBS_LEAK=1e-3,
                 EPSILON=1e-3,
                 ref_prob='unif',
                 augmentation=True,
                 KL_centering=True,
                 rtg_centering=True,
                 do_PG=False,        # Policy gradient on KL update
                 do_intrinsic=False, # interpret KL as an intrinsic reward 
                 monte_carlo=False,
                 explo_drive=True):
        
        self.agent = agent
        try:
            assert agent.isTime == False
        except:
            print('The agent must be a State Agent (isTime=False)')   
        # PARAMETERS
        self.OBS_LEAK = OBS_LEAK
        self.EPSILON = EPSILON
        self.HIST_HORIZON = agent.HIST_HORIZON
        self.ref_prob = ref_prob
        self.augmentation = augmentation
        self.KL_centering = KL_centering
        self.rtg_centering = rtg_centering        
        self.monte_carlo = monte_carlo
        self.do_PG = do_PG            # Policy gradient on KL update
        self.do_intrinsic = do_intrinsic
        self.explo_drive = explo_drive
            
        # SESSION INITIALIZATION
        self.nb_trials = 0
        self.init_trial(update=False)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        
        # SESSION RECORDING
        self.mem_obs_final = []
        self.mem_total_reward = []
        self.mem_total_reward_test = []
        self.mem_mean_rtg = []
        self.mem_KL_final = []
        self.mem_t_final = []
        self.mem_t_final_test = []        

    def init_trial(self, update=True):
        if update:
            self.nb_trials += 1
        self.total_reward = 0
        self.trajectory = []
        self.action_history = []
        self.reward_history = []
        self.rtg_history = []
        self.ktg_history = []
        
    def calc_final_state_probs(self):
        return self.obs_score_final / np.sum(self.obs_score_final)
        
    def calc_ref_probs(self, obs, EPSILON=1e-3):
        if self.ref_prob == 'unif':
            # EXPLORATION DRIVE
            if self.augmentation:
                p = np.zeros(self.agent.N_obs)
                p[np.where(self.nb_visits_final > 0)] = 1 / np.sum(self.nb_visits_final > 0)              
            else:
                p = np.ones(self.agent.N_obs) / self.agent.N_obs
            return p
        else:
            # SET POINT
            ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
            ref_probs[int(self.ref_prob)] += (1 - EPSILON)
            return ref_probs
    
    def KL(self, obs, done=False, verbose=False):
        if done:
            final_state_probs = self.calc_final_state_probs()
            ref_probs = self.calc_ref_probs(obs, EPSILON=self.EPSILON)
            return np.log(final_state_probs[obs]) - np.log(ref_probs[obs])
        else:
            return 0   
        
    def calc_sum_future_KL(self, obs, done):
        sum_future_KL = self.KL(obs, done=done)
        if not done:
            next_values = self.agent.set_Q_obs(obs, Q=self.agent.Q_KL)
            next_sum = self.agent.softmax_expectation(obs, next_values)
            sum_future_KL += self.agent.GAMMA * next_sum
        if self.KL_centering:
            sum_future_KL -= np.mean(self.agent.Q_KL_tab)
        return sum_future_KL
    
    def calc_TD_err_KL(self, sum_future_KL, past_obs, past_action):
        return 1 / self.agent.BETA * (0.01 * self.agent.Q_KL(past_obs, past_action) +  sum_future_KL)
    
    def online_KL_err(self, past_obs, past_action, obs, done=False):
        sum_future_KL = self.calc_sum_future_KL(obs, done)
        return self.agent.Q_KL(past_obs, past_action) - sum_future_KL
    
    def calc_sum_future_rewards(self, reward, obs, done, actions_set=None):
        sum_future_rewards = reward
        if not done:
            next_values = self.agent.set_Q_obs(obs, Q=self.agent.Q_ref)
            next_sum = self.agent.softmax_expectation(obs, next_values)
            sum_future_rewards += self.agent.GAMMA * next_sum 
        if self.rtg_centering:
            sum_future_rewards -= np.mean(self.agent.Q_ref_tab)
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs, past_action):
        return self.agent.PREC * (self.agent.Q_ref(past_obs, past_action) - sum_future_rewards)
    
    def online_TD_err_ref(self, past_obs, past_action, obs, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs, past_action)

    def calc_ELBO(self, sum_future_KL, past_obs, past_action):     
        # !! TEST 4
        return  - 0.5 * self.agent.PREC *(self.agent.Q_var(past_obs, past_action) - self.agent.Q_ref(past_obs, past_action)) **2 \
        - int(self.explo_drive) / self.agent.BETA * sum_future_KL
        #return  - 0.5 * self.agent.PREC /self.agent.BETA * (self.agent.Q_var(past_obs, past_action) - sum_future_rewards) **2 \
    def online_ELBO(self, past_obs, past_action, obs, reward, done=False):      
        sum_future_KL = self.agent.Q_KL(past_obs, past_action)        
        return self.calc_ELBO(sum_future_KL, past_obs, past_action)    
     
    def calc_TD_err_var(self, sum_future_rewards, past_obs, past_action):
        return   self.agent.PREC * (self.agent.Q_var(past_obs, past_action) - sum_future_rewards )
        #return (self.agent.Q_var(past_obs, past_action) - self.agent.PREC * sum_future_rewards)
    
    def online_TD_err_var(self, past_obs, past_action, obs, reward, done=False):      
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done)        
        return self.calc_TD_err_var(sum_future_rewards, past_obs, past_action)
    
    def online_update(self, past_obs, past_action, obs, reward, done):
        
        self.agent.Q_KL_tab[past_obs, past_action] -= self.agent.ALPHA * self.online_KL_err(past_obs,
                                                                                          past_action,
                                                                                          obs,
                                                                                          done=done)
        if self.agent.do_reward:
            self.agent.Q_ref_tab[past_obs, past_action] -= self.agent.ALPHA * self.online_TD_err_ref(past_obs,
                                                                                              past_action,
                                                                                              obs,
                                                                                              reward,
                                                                                              done=done)
        TD_err_var = self.online_TD_err_var(past_obs,past_action,obs,reward,done=done)
        ELBO = self.online_ELBO(past_obs,past_action,obs,reward,done=done)
        self.agent.Q_var_tab[past_obs, past_action] += self.agent.ALPHA  * self.agent.Q_VAR_MULT * (ELBO - TD_err_var)
        
        if self.do_PG:
            pi = self.agent.softmax(past_obs)
            #sum_future_KL = self.agent.Q_KL(past_obs, past_action)        
            for a in range(self.agent.N_act):
                self.agent.Q_var_tab[past_obs, a] -= self.agent.ALPHA * self.agent.Q_VAR_MULT * pi[a] * ELBO #/ self.agent.BETA * sum_future_KL

    def monte_carlo_update(self, done):
        final_time = self.agent.get_time()
        liste_sum_KL = np.zeros(final_time)
        liste_rtg = np.zeros(final_time+1)            

        #  KL_to_go calculation
        new_obs = self.trajectory[final_time]
        final_KL = self.KL(new_obs, done=done) 
        liste_sum_KL  = final_KL * self.agent.GAMMA **(final_time - np.arange(1, final_time+1))
        self.ktg_history.append(liste_sum_KL)
        if self.KL_centering:
            mean_KL_final = np.mean(self.mem_KL_final[-100:])
            
        # Reward_to_go calculation
        if self.agent.do_reward:
            for time in range(final_time):
                liste_rtg[time] = np.sum(np.array(self.reward_history[time:]) * \
                                      self.agent.GAMMA **(np.arange(time, final_time) - time))
                self.rtg_history.append(liste_rtg[time])      
            if self.rtg_centering:
                mean_mean_rtg = np.mean(self.mem_mean_rtg[-100:])

        # MAIN LOOP
        for time in range(final_time):                
            past_obs = self.trajectory[time]
            past_time = time                
            past_action = self.action_history[time]
            sum_future_KL = liste_sum_KL[time]  #
            if self.KL_centering:
                sum_future_KL -= mean_KL_final
            sum_future_rewards = liste_rtg[time] 
            if self.agent.do_reward and self.rtg_centering:
                sum_future_rewards -= mean_mean_rtg

            if self.nb_trials > 10:
                TD_err_KL = self.calc_TD_err_KL(sum_future_KL, past_obs, past_action)
                self.agent.Q_KL_tab[past_time, past_action] -= self.agent.ALPHA * TD_err_KL
                
                TD_err_ref = self.calc_TD_err_ref(sum_future_rewards, past_obs, past_action)
                self.agent.Q_ref_tab[past_obs, past_action] -= self.agent.ALPHA * TD_err_ref

                ELBO = self.calc_ELBO(sum_future_KL,
                                      past_obs,
                                      past_action)
                TD_err_var = self.calc_TD_err_var(sum_future_rewards, past_obs, past_action)
                
                self.agent.Q_var_tab[past_obs, past_action] += self.agent.ALPHA * (ELBO - TD_err_var)
                
                if self.do_PG:
                    pi = self.agent.softmax(past_obs)
                    for a in range(self.agent.N_act):
                        self.agent.Q_var_tab[past_obs, a] -= self.agent.ALPHA *  pi[a] * ELBO  #/ self.agent.BETA * sum_future_KL

    def run_episode(self, train=True, render=False, verbose=False):
        self.agent.init_env()
        self.init_trial(update=train)
        obs = self.agent.get_observation()
        self.trajectory.append(obs)
        tic = time.clock()

        while True:
            
            past_time = self.agent.get_time()
            past_obs = obs 

            ########### STEP #############
            if train:
                past_obs, past_action, obs, reward, done = self.agent.step()
            else:
                past_obs, past_action, obs, reward, done = self.agent.step(test=True)
            ##############################

            current_time = self.agent.get_time()
            obs = self.agent.get_observation()

            self.action_history.append(past_action)
            self.trajectory.append(obs)
            
            self.agent.num_episode += 1

            if done:
                self.nb_visits_final[obs] += 1
                self.obs_score_final *= 1 - self.OBS_LEAK
                self.obs_score_final[obs] += 1
                self.mem_obs_final += [obs]

            mem_reward = reward
            if not self.agent.do_reward:
                reward = 0
            self.reward_history.append(reward)
            self.total_reward += mem_reward       

            if train and not self.monte_carlo:
                ########### LEARNING STEP #############
                self.online_update(past_obs, past_action, obs, reward, done)                
                #######################################
            if done:
                if train:
                    if self.monte_carlo:
                        ########### LEARNING STEP #############
                        self.monte_carlo_update(done)                
                        #######################################
                    KL_final = self.KL(obs, done=done)
                    self.mem_KL_final.append(KL_final)
                    self.mem_t_final.append(current_time)
                    self.mem_total_reward.append(self.total_reward)
                    self.mem_mean_rtg.append(np.mean(self.rtg_history))
                    if verbose:
                        print('obs:', obs, 'final KL loss:', KL_final) 
                else:
                    self.mem_t_final_test.append(current_time)
                    self.mem_total_reward_test.append(self.total_reward)
                break
        toc = time.clock()
        if verbose:
            print('Time elapsed :', toc-tic)


