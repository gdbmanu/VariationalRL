import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma
from environment import Environment
from agent import Transition
from sklearn.neighbors import KernelDensity

import torch
import time


class Trainer():

    def __init__(self, agent,
                 OBS_LEAK=1e-3,
                 EPSILON=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 augmentation=True,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 KL_centering=True,
                 rtg_centering=True):
        self.agent = agent
        self.nb_trials = 0
        self.init_trial(update=False)
        self.nb_visits = np.zeros(self.agent.env.N_obs)
        self.obs_score = np.zeros(self.agent.env.N_obs)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        self.mem_obs_final = []
        self.mem_total_reward = []
        self.mem_total_reward_test = []
        self.mem_mean_rtg = []
        self.mem_KL_final = []
        self.mem_t_final = []
        self.mem_t_final_test = []
        self.OBS_LEAK = OBS_LEAK
        self.EPSILON = EPSILON
        self.HIST_HORIZON = agent.HIST_HORIZON
        self.mem_V = {}
        self.ref_prob = ref_prob
        self.final = final
        self.monte_carlo = monte_carlo
        self.Q_learning = Q_learning
        self.KL_reward = KL_reward
        self.augmentation = augmentation
        self.KL_correction = KL_correction
        self.Q_ref_correction = Q_ref_correction
        self.BATCH_SIZE=BATCH_SIZE
        self.KL_centering = KL_centering
        self.rtg_centering = rtg_centering

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

    def calc_state_probs(self):
        self.obs_score / np.sum(self.obs_score)
        
    def calc_final_state_probs(self):
        return self.obs_score_final / np.sum(self.obs_score_final)
        
    def calc_ref_probs(self, obs, EPSILON=1e-3):
        if self.ref_prob == 'unif':
            # EXPLORATION DRIVE
            if self.augmentation:
                p = np.zeros(self.agent.N_obs)
                if self.final:
                    p[np.where(self.nb_visits_final > 0)] = 1 / np.sum(self.nb_visits_final > 0)
                else:
                    p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0)
            else:
                p = np.ones(self.agent.N_obs) / self.agent.N_obs
            return p
        else:
            # SET POINT
            ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
            ref_probs[int(self.ref_prob)] += (1 - EPSILON)
            return ref_probs
    
    def KL(self, obs, done=False, verbose=False):
        if self.final: # Only final state for probability calculation
            if done:
                final_state_probs = self.calc_final_state_probs()
                ref_probs = self.calc_ref_probs(obs, EPSILON=self.EPSILON)
                return np.log(final_state_probs[obs]) - np.log(ref_probs[obs])
            else:
                return 0
        else:
            state_probs = self.calc_state_probs()
            ref_probs = self.calc_ref_probs(obs, EPSILON=self.EPSILON)
            return np.log(state_probs[obs]) - np.log(ref_probs[obs])  # +1


    def calc_sum_future_KL(self, obs, obs_or_time, done, actions_set=None):
        sum_future_KL = self.KL(obs, done=done)
        if not done:
            next_values = self.agent.set_Q_obs(obs_or_time,
                                               Q=self.agent.Q_KL,
                                               actions_set=actions_set)
            next_sum = self.agent.softmax_expectation(obs_or_time,
                                                      next_values,
                                                      actions_set=actions_set)
            sum_future_KL += self.agent.GAMMA * next_sum
        return sum_future_KL


    def online_KL_err(self, past_obs_or_time, past_action, obs, obs_or_time, done=False):
        sum_future_KL = self.calc_sum_future_KL(obs, obs_or_time, done)
        return self.agent.Q_KL(past_obs_or_time, past_action) - sum_future_KL


    def calc_sum_future_rewards(self, reward, obs_or_time, done, actions_set=None):
        sum_future_rewards = reward
        if not done:
            next_values = self.agent.set_Q_obs(obs_or_time,
                                               Q=self.agent.Q_ref,
                                               actions_set=actions_set)
            next_sum = self.agent.softmax_expectation(obs_or_time,
                                                        next_values,
                                                        actions_set=actions_set)
            sum_future_rewards += self.agent.GAMMA * next_sum
        
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs_or_time, past_action):
        return self.agent.PREC * (self.agent.Q_ref(past_obs_or_time, past_action) - sum_future_rewards)

    def online_TD_err_ref(self, past_obs_or_time, past_action, obs_or_time, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs_or_time, done)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)


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


    def online_update(self, past_obs, past_action, obs, reward, done, past_time, current_time, actions_set=None):
        if self.agent.isTime:
            past_obs_or_time = past_time
            obs_or_time = current_time
        else:
            past_obs_or_time = past_obs
            obs_or_time = obs
        if not self.Q_learning:
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
                    liste_sum_KL[time]  = np.sum(np.array(liste_KL[time:]) * \
                                           self.agent.GAMMA **(np.arange(time, final_time+1) - time))
                    self.ktg_history.append(liste_sum_KL[time])
                    
            if self.agent.do_reward:
                for time in range(final_time):
                    liste_rtg[time] = np.sum(np.array(self.reward_history[time:]) * \
                                          self.agent.GAMMA **(np.arange(time, final_time) - time))
                    self.rtg_history.append(liste_rtg[time])
                    
            # THIRD LOOP
            if self.KL_centering:
                mean_KL_final = np.mean(self.mem_KL_final[-100:])
            if self.rtg_centering:
                mean_mean_rtg = np.mean(self.mem_mean_rtg[-100:])
            for time in range(final_time):                ## !!!! faux dans le cas "full KL" et "full reward" !!!! TODO ##
                past_obs = self.trajectory[time]
                if self.agent.isTime:
                    past_obs_or_time = time
                else:
                    past_obs_or_time = self.trajectory[time]
                past_action = self.action_history[time]
                sum_future_KL = liste_sum_KL[time]  #
                if self.KL_centering:
                    sum_future_KL -= mean_KL_final
                sum_future_rewards = liste_rtg[time] 
                if self.rtg_centering:
                    sum_future_rewards -= mean_mean_rtg

                if self.nb_trials > 10:
                    TD_err_ref = self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)
                    self.agent.Q_ref_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * TD_err_ref

                    TD_err_var = self.calc_TD_err_var(sum_future_rewards,
                                                      sum_future_KL ,
                                                      past_obs,
                                                      past_obs_or_time,
                                                      past_action)
                    self.agent.Q_var_tab[past_obs_or_time, past_action] -= self.agent.ALPHA * TD_err_var




    def run_episode(self, train=True, render=False, verbose=False):
        self.agent.init_env()
        self.init_trial(update=train)
        obs = self.agent.get_observation()
        self.trajectory.append(obs)
        tic = time.clock()

        while True:
            
            past_time = self.agent.get_time()
            past_obs = obs #self.agent.get_observation()
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
            
            self.nb_visits[obs] += 1
            self.obs_score *= 1 - self.OBS_LEAK
            self.obs_score[obs] += 1
            self.agent.num_episode += 1


            if done:
                self.nb_visits_final[obs] += 1
                self.obs_score_final *= 1 - self.OBS_LEAK
                self.obs_score_final[obs] += 1
                self.mem_obs_final += [obs]

            if past_time == 0 and train:
                self.state_probs = self.calc_state_probs()
                self.ref_probs = self.calc_ref_probs(obs)

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
                    if self.nb_trials % 100 == 0 and not self.agent.isTime:
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
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=False,
                 augmentation=False,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 KL_centering=True,
                 rtg_centering=True):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         KL_centering=KL_centering,
                         rtg_centering=rtg_centering)


class KL_Q_learning_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=True,
                 augmentation=True,
                 KL_correction=False,
                 Q_ref_correction=False,                 
                 BATCH_SIZE=20,
                 KL_centering=True,
                 rtg_centering=True):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         KL_centering=KL_centering,
                         rtg_centering=rtg_centering)


class One_step_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 augmentation=True,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 KL_centering=True,
                 rtg_centering=True):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=False,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         KL_centering=KL_centering,
                         rtg_centering=rtg_centering)


class Final_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 augmentation=True,
                 KL_correction=False,
                 Q_ref_correction=False,
                 BATCH_SIZE=20,
                 KL_centering=True,
                 rtg_centering=True):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=Q_learning,
                         KL_reward=KL_reward,
                         augmentation=augmentation,
                         KL_correction=KL_correction,
                         Q_ref_correction=Q_ref_correction,
                         BATCH_SIZE=BATCH_SIZE,
                         KL_centering=KL_centering,
                         rtg_centering=rtg_centering)


