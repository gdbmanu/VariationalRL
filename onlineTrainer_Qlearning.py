import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma
from environment import Environment
from sklearn.neighbors import KernelDensity

import torch
import time


class OnlineTrainer():

    def __init__(self, agent,
                 augmentation=True,
                 monte_carlo=False,
                 rtg_centering=True):
        
        self.agent = agent
        try:
            assert agent.isTime == False
        except:
            print('The agent must be a State Agent (isTime=False)')   
            
        # SESSION INITIALIZATION
        self.nb_trials = 0
        self.init_trial(update=False)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        
        self.monte_carlo = monte_carlo
        self.rtg_centering =rtg_centering        
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
    
    def calc_sum_future_rewards(self, reward, obs, done, actions_set=None):
        sum_future_rewards = reward
        if not done:
            next_Q = self.agent.set_Q_obs(obs, Q=self.agent.Q_ref)
            next_sum = self.agent.greedy_act_max(next_Q)
            sum_future_rewards += self.agent.GAMMA * next_sum 
        if self.rtg_centering:
            sum_future_rewards -= np.mean(self.agent.Q_ref_tab)
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs, past_action):
        return self.agent.Q_ref(past_obs, past_action) - sum_future_rewards
    
    def online_TD_err_ref(self, past_obs, past_action, obs, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs, past_action)
    
    def online_update(self, past_obs, past_action, obs, reward, done):
                
        if self.agent.do_reward:
            self.agent.Q_ref_tab[past_obs, past_action] -= self.agent.ALPHA * self.online_TD_err_ref(past_obs,
                                                                                              past_action,
                                                                                              obs,
                                                                                              reward,
                                                                                              done=done)

    def monte_carlo_update(self, done):
        final_time = self.agent.get_time()
        liste_rtg = np.zeros(final_time+1)            
 
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
            sum_future_rewards = liste_rtg[time] 
            if self.agent.do_reward and self.rtg_centering:
                sum_future_rewards -= mean_mean_rtg

            if self.nb_trials > 10:
                
                TD_err_ref = self.calc_TD_err_ref(sum_future_rewards, past_obs, past_action)
                self.agent.Q_ref_tab[past_obs, past_action] -= self.agent.ALPHA * TD_err_ref

                

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


