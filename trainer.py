import numpy as np


class Trainer():

    def __init__(self, agent,
                 OBS_LEAK=1e-3,
                 EPSILON=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi = False):
        self.agent = agent
        self.nb_trials = 0
        self.init_trial(update=False)
        self.nb_visits = np.zeros(self.agent.env.N_obs)
        self.obs_score = np.zeros(self.agent.env.N_obs)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        self.OBS_LEAK = OBS_LEAK
        self.EPSILON = EPSILON
        self.mem_V = {}
        self.ref_prob = ref_prob
        self.final = final
        self.monte_carlo = monte_carlo
        self.Q_learning = Q_learning
        self.KL_reward = KL_reward
        self.ignore_pi = ignore_pi

    def init_trial(self, update=True):
        if update:
            self.nb_trials += 1
        self.total_reward = 0
        self.trajectory = []
        self.action_history = []
        self.reward_history = []

    def calc_state_probs(self):
        # return self.nb_visits/np.sum(self.nb_visits)
        return self.obs_score / np.sum(self.obs_score)

    def calc_final_state_probs(self):
        # return self.nb_visits/np.sum(self.nb_visits)
        return self.obs_score_final / np.sum(self.obs_score_final)

    def calc_ref_probs(self, obs, EPSILON=1e-3):

        # state_probs = np.ones(self.agent.env.N_obs) * EPSILON
        # for a in range(self.agent.env.N_act):
        #    if Environment.direction[a] in Environment.next[obs]:
        #        state_probs[Environment.next[obs][Environment.direction[a]]] = 1
        # return state_probs / sum(state_probs)

        if self.ref_prob == 'unif':
            # EXPLORATION DRIVE
            p = np.zeros(self.agent.env.N_obs)
            p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0)
            return p
        else:
            # SET POINT
            ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
            ref_probs[int(self.ref_prob)] += (1 - EPSILON)
            return ref_probs

        # state_probs = np.zeros(self.agent.env.N_obs)
        # for o in range(self.agent.env.N_obs):
        #     state_probs[o] = 7 - o
        # return state_probs / sum(state_probs)

    # agent.KL_diff update
    # def KL(self, past_obs, a, new_obs, done=False):
    #    return 0

    def KL(self, final_obs, done=False):
        if self.final: # Only final state for probability calculation
            if done:
                state_probs = self.calc_final_state_probs()
                ref_probs = self.calc_ref_probs(final_obs, EPSILON=self.EPSILON)
                return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])  # +1
            else:
                return 0  # self.agent.KL[past_obs, a]
        else:
            state_probs = self.calc_state_probs()
            ref_probs = self.calc_ref_probs(final_obs, EPSILON=self.EPSILON)
            return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])

    def online_KL_err(self, past_obs, past_action, obs, obs_or_time, done=False):
        # For policy update
        sum_future_KL = self.KL(obs, done=done)
        if not done:
            next_sum = self.agent.softmax_expectation(obs_or_time, Q=self.agent.KL[obs, :])
            sum_future_KL += self.agent.GAMMA * next_sum                                                            
        return sum_future_KL - self.agent.KL[past_obs, past_action]

            # if current_time is None:
            #     return self.KL(past_obs, past_action, obs, done=done) + \
            #            self.agent.GAMMA * self.agent.softmax_expectation(obs,
            #                                                              Q=self.agent.KL[obs, :]) - \
            #            self.agent.KL[past_obs, past_action]
            # else:
            #     return self.agent.GAMMA * self.agent.softmax_expectation(current_time,
            #                                                              Q=self.agent.KL[obs, :]) - \
            #            self.agent.KL[past_obs, past_action]
            
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time=None):
        return 0
    
    
    def calc_sum_future_rewards(self, reward, done, obs_or_time):
        sum_future_rewards = reward
        if not done:
            Q=self.agent.Q_ref[obs_or_time, :]
            sum_future_rewards += self.agent.GAMMA * self.agent.softmax_expectation(obs_or_time, Q)
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs_or_time, past_action):
        return self.agent.BETA * (sum_future_rewards - self.agent.Q_ref[past_obs_or_time, past_action])
        
    def online_TD_err_ref(self, past_obs_or_time, past_action, obs_or_time, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, done, obs_or_time)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)

    
    def calc_TD_err_var(self, sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        pi = self.agent.softmax(past_obs_or_time, Q = self.agent.Q_ref)[past_action]
        #pi = self.agent.softmax(past_obs_or_time, Q = self.agent.Q_var)[past_action]
        mult_pi = 1 - pi # 1 # 
        return self.agent.BETA * (sum_future_rewards - self.agent.Q_var[past_obs_or_time, past_action] \
               - mult_pi *  mult_Q * sum_future_KL)
               #- mult_Q / self.agent.BETA * sum_future_KL 

    def online_TD_err_var(self, past_obs, past_obs_or_time, past_action, obs, obs_or_time, reward, done=False):      
        sum_future_rewards = self.calc_sum_future_rewards(reward, done, obs_or_time)
        sum_future_KL = self.agent.KL[past_obs, past_action] 
        return self.calc_TD_err_var(sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action)
        
            #return self.agent.BETA * (reward - self.agent.Q_var[past_obs_or_time, past_action]) \
            #       - mult_Q * self.KL_diff(past_obs, past_action, obs, past_time=past_time)    
        
    
                #!! self.KL_diff(past_obs, past_action, obs, past_time=past_time)
            
            # return self.agent.BETA * (
            #                 reward +
            #                 self.agent.GAMMA * self.agent.softmax_expectation(obs_or_time, Q=Q_ref) -
            #                 Q_var
            #             ) \
            #       - mult_Q * self.KL_diff(past_obs, past_action, obs, done=done, past_time=past_time)
      

    def BETA_err(self, past_obs, past_action, past_time=None):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        if self.agent.isTime:
            past_obs_or_time = past_time
        else:
            past_obs_or_time = past_obs
        return - mult_Q * (self.agent.Q_var[past_obs_or_time, past_action]
                           - self.agent.softmax_expectation(past_obs_or_time)) \
               * self.agent.KL[past_obs, past_action]

    def online_update(self, past_obs, past_action, obs, reward, done, past_time, current_time):
        if self.agent.isTime:
            past_obs_or_time = past_time
            obs_or_time = current_time
        else:
            past_obs_or_time = past_obs
            obs_or_time = obs
        if not self.Q_learning:
            self.agent.KL[past_obs, past_action] += self.agent.ALPHA * 30 * self.online_KL_err(past_obs,
                                                                                          past_action,
                                                                                          obs,
                                                                                          obs_or_time,
                                                                                          done=done)
        self.agent.Q_ref[past_obs_or_time, past_action] += self.agent.ALPHA * self.online_TD_err_ref(past_obs_or_time,
                                                                                              past_action,
                                                                                              obs_or_time,
                                                                                              reward,
                                                                                              done=done)
        self.agent.Q_var[past_obs_or_time, past_action] += self.agent.ALPHA * self.online_TD_err_var(past_obs,
                                                                                              past_obs_or_time,
                                                                                              past_action,
                                                                                              obs,
                                                                                              obs_or_time,
                                                                                              reward,
                                                                                              done=done)
        # self.agent.BETA += self.agent.ALPHA * 0.1 * self.BETA_err(past_obs,
        #                                                            past_action,
        #                                                            obs)


    def monte_carlo_update(self, done):
        if done:
            current_time = self.agent.env.get_time()
            liste_KL = np.zeros(current_time)
            liste_reward = np.zeros(current_time)
            for time in range(current_time):
                new_obs = self.trajectory[time + 1]
                test_done = current_time == time + 1
                liste_KL[time] = self.KL(new_obs, done=test_done) * self.agent.GAMMA ** (current_time - time + 1)
                liste_reward[time] = self.reward_history[time] * self.agent.GAMMA ** (current_time - time + 1)
            for time in range(current_time):
                past_obs = self.trajectory[time]
                if self.agent.isTime:
                    past_obs_or_time = time
                else:
                    past_obs_or_time = self.trajectory[time]
                past_action = self.action_history[time]

                

                # TD_err_ref = self.agent.BETA * (
                #                 np.sum(liste_reward[time:])
                #                 - self.agent.Q_ref[past_obs_or_time, past_action]
                #                 )
                TD_err_ref = self.calc_TD_err_ref(np.sum(liste_reward[time:]), past_obs_or_time, past_action)
                            #np.sum(liste_reward[time:]) \
                            # - self.agent.Q_ref[past_obs_or_time, past_action]
                self.agent.Q_ref[past_obs_or_time, past_action] += self.agent.ALPHA * TD_err_ref

                #if self.ignore_pi:
                mult_pi = 1
                #else:
                #    mult_pi = 1 - pi

                # TD_err_var = self.agent.BETA * (
                #                 np.sum(liste_reward[time:])
                #                 - self.agent.Q_var[past_obs_or_time, past_action]
                #                 #- mult_Q *  mult_pi * np.sum(liste_KL[time:])
                #                 - mult_Q * np.sum(liste_KL[time:])
                # )
                TD_err_var = self.calc_TD_err_var(np.sum(liste_reward[time:]), 
                                                  np.sum(liste_KL[time:]),
                                                  past_obs, 
                                                  past_obs_or_time, 
                                                  past_action)
        
                
                #np.sum(liste_reward[time:]) \
                #        - self.agent.Q_var[past_obs_or_time, past_action] \
                #        - mult_Q / self.agent.BETA * mult_pi * np.sum(liste_KL[time:])
                
                # - mult_Q *  mult_pi * np.sum(liste_KL[time:])

                # diff_Q = np.sum(liste_reward[time:]) - self.agent.Q_var[past_obs_or_time, past_action]
                # TD_err_var =  diff_Q + mult_Q * mult_pi * self.agent.BETA * (- diff_Q**2 -  np.sum(liste_KL[time:]))

                self.agent.Q_var[past_obs_or_time, past_action] += self.agent.ALPHA * TD_err_var

                # BETA_err = - mult_Q * (self.agent.Q_var[time, past_action]
                #            - self.agent.softmax_expectation(time)) \
                #            * np.sum(liste_KL[time:])
                # self.agent.BETA += self.agent.ALPHA * 0.3 * BETA_err


    def run_episode(self):
        self.agent.init_env()
        self.init_trial()
        self.trajectory.append(self.agent.env.get_observation())

        while True:

            past_time = self.agent.env.get_time()
            past_obs = self.agent.env.get_observation()
            past_obs_or_time, past_action, obs_or_time, reward, done = self.agent.step()
            current_time = self.agent.env.get_time()
            obs = self.agent.env.get_observation()

            self.action_history.append(past_action)
            self.trajectory.append(obs)

            self.nb_visits[obs] += 1
            self.obs_score *= 1 - self.OBS_LEAK
            self.obs_score[obs] += 1
            self.agent.num_episode += 1
            if done:
                self.nb_visits_final[obs] += 1
                self.obs_score_final *= 1 - self.OBS_LEAK
                self.obs_score_final[obs] += 1

            mem_reward = reward
            if not self.agent.do_reward:
                reward = 0
            if self.KL_reward:
                reward -= self.KL(obs, done=done)
            self.reward_history.append(reward)
            self.total_reward += mem_reward

            if self.monte_carlo:
                self.monte_carlo_update(done)
            else:
                self.online_update(past_obs, past_action, obs, reward, done, past_time, current_time)

            if done:
                if self.nb_trials % 100 == 0 and not self.agent.isTime:
                    V = np.zeros(self.agent.env.N_obs)
                    for s in range(self.agent.env.N_obs):
                        V[s] = self.agent.softmax_expectation(s)
                    self.mem_V[self.nb_trials] = V
                break


class Q_learning_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward)


class KL_Q_learning_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 KL_reward=True):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=True,
                         KL_reward=KL_reward)


class One_step_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi = False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=False,
                         KL_reward=KL_reward,
                         ignore_pi=ignore_pi)

    # agent.Q_var update
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time=None):
        pi = self.agent.softmax(past_obs)[a]
        state_probs = self.agent.calc_state_probs(past_obs)
        ref_probs = self.calc_ref_probs(past_obs, EPSILON=self.EPSILON)
        return (1 - pi) * self.agent.BETA * (np.log(state_probs[new_obs]) + pi / state_probs[new_obs] * 1 - np.log(ref_probs[new_obs]))


class Final_variational_trainer(Trainer):

    def __init__(self, agent,
                 EPSILON=1e-3,
                 OBS_LEAK=1e-3,
                 ref_prob='unif',
                 final=False,
                 monte_carlo=False,
                 Q_learning=False,
                 KL_reward=False,
                 ignore_pi = False):
        super().__init__(agent,
                         EPSILON=EPSILON,
                         OBS_LEAK=OBS_LEAK,
                         ref_prob=ref_prob,
                         final=final,
                         monte_carlo=monte_carlo,
                         Q_learning=Q_learning,
                         KL_reward=KL_reward,
                         ignore_pi=ignore_pi)

    # agent.Q_var update
    def KL_diff(self, past_obs, a, final_obs, done=False, past_time=None):
        if past_time is None:
            pi = self.agent.softmax(past_obs)[a]
        else:
            pi = self.agent.softmax(past_time)[a]
        if self.ignore_pi:
            return self.agent.BETA * self.agent.KL[past_obs, a]
        else:
            return (1 - pi) * self.agent.BETA * self.agent.KL[past_obs, a]  # self.KL(past_obs, a, final_obs, done)
