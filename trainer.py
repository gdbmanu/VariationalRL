import numpy as np
from scipy.stats import multivariate_normal
import torch

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
        if self.agent.isDiscrete:
            self.nb_visits = np.zeros(self.agent.env.N_obs)
            self.obs_score = np.zeros(self.agent.env.N_obs)
            self.nb_visits_final = np.zeros(self.agent.env.N_obs)
            self.obs_score_final = np.zeros(self.agent.env.N_obs)
        else:
            self.mem_obs = []
            self.mem_obs_final = []
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
        if self.agent.isDiscrete:
            return self.obs_score / np.sum(self.obs_score)
        else:
            b_inf = min(1000, len(self.mem_obs))
            mu = np.mean(self.mem_obs[-b_inf:], axis = 0)
            Sigma = np.cov(np.array(self.mem_obs[-b_inf:]).T)
            try:
                rv = multivariate_normal(mu, Sigma)
            except:
                try:
                    var = np.var(trainer.mem_obs_final, axis=0)
                    rv = multivariate_normal(mu, var)
                except:
                    rv = multivariate_normal(mu, np.ones(len(mu)))
            return rv.pdf # !! TODO a verifier  

    def calc_final_state_probs(self):
        # return self.nb_visits/np.sum(self.nb_visits)
        if self.agent.isDiscrete:
            return self.obs_score_final / np.sum(self.obs_score_final)
        else:
            mu = np.mean(self.mem_obs_final, axis=0)
            Sigma = np.cov(np.array(self.mem_obs_final).T)
            try:
                rv = multivariate_normal(mu, Sigma)
            except:
                try:
                    var = np.var(trainer.mem_obs_final, axis=0)
                    rv = multivariate_normal(mu, var)
                except:
                    rv = multivariate_normal(mu, np.ones(len(mu)))          
            return rv.pdf # !! TODO a verifier 

    def calc_ref_probs(self, obs, EPSILON=1e-3, final=False):
        if self.ref_prob == 'unif':
            # EXPLORATION DRIVE
            if self.agent.isDiscrete:
                p = np.zeros(self.agent.N_obs)
                if self.final:
                    ## !!!! A revoir TODO !!!!!!!!
                    #p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0) 
                    p[np.where(self.nb_visits_final > 0)] = 1 / np.sum(self.nb_visits_final > 0)
                else:
                    p[np.where(self.nb_visits > 0)] = 1 / np.sum(self.nb_visits > 0)
                return p
            else:
                return 1 / np.prod(self.agent.env.observation_space.high - self.agent.env.observation_space.low)
        else:
            # SET POINT
            if self.agent.isDiscrete:
                ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
                ref_probs[int(self.ref_prob)] += (1 - EPSILON)
                return ref_probs
            else:
                return None # TODO

    def KL(self, final_obs, done=False):
        if self.final: # Only final state for probability calculation
            if done:
                final_state_probs = self.calc_final_state_probs()
                ref_probs = self.calc_ref_probs(final_obs, EPSILON=self.EPSILON)
                if self.agent.isDiscrete:
                    return np.log(final_state_probs[final_obs]) - np.log(ref_probs[final_obs])  # +1
                else:
                    print('obs :', final_obs, ', KL loss : ', np.log(final_state_probs(final_obs)) - np.log(ref_probs))
                    return np.log(final_state_probs(final_obs)) - np.log(ref_probs)  # +1
            else:
                return 0  # self.agent.KL_tab[past_obs, a]
        else:
            state_probs = self.calc_state_probs()
            ref_probs = self.calc_ref_probs(final_obs, EPSILON=self.EPSILON)
            if self.agent.isDiscrete:
                return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])  # +1
            else:
                if done:
                    print('obs :', final_obs, ', KL loss : ', np.log(state_probs(final_obs)) - np.log(ref_probs))
                return np.log(state_probs(final_obs)) - np.log(ref_probs)  # +1

    def calc_sum_future_KL(self, obs, obs_or_time, done, tf=False):
        sum_future_KL = self.KL(obs, done=done)
        if tf:
            sum_future_KL = torch.FloatTensor([sum_future_KL])      
        if not done:
            next_values = self.agent.set_Q_obs(obs, Q=self.agent.KL, tf=tf)
            next_sum = self.agent.softmax_expectation(obs, next_values, tf=tf)
            sum_future_KL += self.agent.GAMMA * next_sum
        return sum_future_KL

    def online_KL_err(self, past_obs, past_action, obs, obs_or_time, done=False):
        # For policy update
        # Only for baseline Environment
        #sum_future_KL = self.KL(obs, done=done)
        #if not done:
        #    next_sum = self.agent.softmax_expectation(obs_or_time, self.agent.KL_tab[obs, :])
        #    sum_future_KL += self.agent.GAMMA * next_sum
        sum_future_KL = self.calc_sum_future_KL(obs, obs_or_time, done)
        return sum_future_KL - self.agent.KL(past_obs, past_action)

    def KL_loss_tf(self, KL_pred_tf, obs, done): # Q TD_error
        sum_future_KL = self.calc_sum_future_KL(obs, obs, done, tf=False) # not tf=True!!!
        sum_future_KL_tf = torch.FloatTensor([sum_future_KL])
        return torch.sum(torch.pow(self.agent.BETA * (sum_future_KL_tf - KL_pred_tf), 2), 1)
            
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time=None):
        return 0

    def calc_sum_future_rewards(self, reward, obs_or_time, done, tf=False):
        sum_future_rewards = reward
        if tf:
            sum_future_rewards = torch.FloatTensor([sum_future_rewards])
        if not done:
            next_values = self.agent.set_Q_obs(obs_or_time, Q=self.agent.Q_ref, tf=tf)
            sum_future_rewards += self.agent.GAMMA * self.agent.softmax_expectation(obs_or_time, next_values, tf=tf)
        
        return sum_future_rewards
    
    def calc_TD_err_ref(self, sum_future_rewards, past_obs_or_time, past_action):
        return self.agent.BETA * (sum_future_rewards - self.agent.Q_ref(past_obs_or_time, past_action))
        
    def online_TD_err_ref(self, past_obs_or_time, past_action, obs_or_time, reward, done=False):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs_or_time, done)
        return self.calc_TD_err_ref(sum_future_rewards, past_obs_or_time, past_action)

    def Q_ref_loss_tf(self, Q_ref_pred_tf, obs, reward, done):
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done)
        sum_future_rewards_tf = torch.FloatTensor([sum_future_rewards])
        return torch.sum(torch.pow(self.agent.BETA * (sum_future_rewards_tf - Q_ref_pred_tf), 2), 1)

    def calc_TD_err_var(self, sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #pi = self.agent.softmax(past_obs_or_time, Q = self.agent.Q_ref)[past_action]
        pi = self.agent.softmax(past_obs_or_time, Q = self.agent.Q_var)[past_action]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mult_pi = 1 - pi # 1 # 
        return self.agent.BETA * (sum_future_rewards - self.agent.Q_var(past_obs_or_time, past_action) \
               - mult_pi *  mult_Q * sum_future_KL)

    def online_TD_err_var(self, past_obs, past_obs_or_time, past_action, obs, obs_or_time, reward, done=False):      
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs_or_time, done)
        sum_future_KL = self.agent.KL(past_obs, past_action)
        return self.calc_TD_err_var(sum_future_rewards, sum_future_KL, past_obs, past_obs_or_time, past_action)

    def Q_var_loss_tf(self, Q_var_pred_tf, KL_pred_tf, obs, reward, done):
        if self.Q_learning:
            mult_Q = 0
        else:
            mult_Q = 1
        sum_future_rewards = self.calc_sum_future_rewards(reward, obs, done, tf=False)
        sum_future_rewards_tf = torch.FloatTensor([sum_future_rewards])
        #sum_future_KL_tf = self.agent.KL(past_obs, past_action, tf=True)
        #sum_future_KL_tf = self.calc_sum_future_KL(past_obs, past_obs, done, tf=True) 
        #sum_future_KL_tf -= torch.FloatTensor([self.KL(past_obs, done=done)])
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #pi = self.agent.softmax(past_obs, Q=self.agent.Q_ref)[past_action]
        #pi = self.agent.softmax(past_obs, Q = self.agent.Q_var)[past_action]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #mult_pi = 1 - pi
        #target_tf = torch.FloatTensor([sum_future_rewards - mult_pi * mult_Q * sum_future_KL])
        #return torch.sum(torch.pow(self.agent.BETA * (target_tf - Q_var_pred_tf), 2), 1)
        return torch.sum(torch.pow(self.agent.BETA * (sum_future_rewards_tf - Q_var_pred_tf), 2) + KL_pred_tf, 1) 
        #return torch.sum(KL_pred_tf)



    def online_update(self, past_obs, past_action, obs, reward, done, past_time, current_time):
        if self.agent.isTime:
            past_obs_or_time = past_time
            obs_or_time = current_time
        else:
            past_obs_or_time = past_obs
            obs_or_time = obs
        if self.agent.isDiscrete:
            if not self.Q_learning:
                self.agent.KL_tab[past_obs, past_action] += self.agent.ALPHA * 30 * self.online_KL_err(past_obs,
                                                                                              past_action,
                                                                                              obs,
                                                                                              obs_or_time,
                                                                                              done=done)
            self.agent.Q_ref_tab[past_obs_or_time, past_action] += self.agent.ALPHA * self.online_TD_err_ref(past_obs_or_time,
                                                                                                  past_action,
                                                                                                  obs_or_time,
                                                                                                  reward,
                                                                                                  done=done)
            self.agent.Q_var_tab[past_obs_or_time, past_action] += self.agent.ALPHA * self.online_TD_err_var(past_obs,
                                                                                                  past_obs_or_time,
                                                                                                  past_action,
                                                                                                  obs,
                                                                                                  obs_or_time,
                                                                                                  reward,
                                                                                                  done=done)
        else:
            if not self.Q_learning:
                KL_pred_tf = self.agent.KL(past_obs, past_action, tf=True)
                #print(KL_pred_tf.detach().numpy(), self.KL(obs, done=done))
                loss_KL = self.KL_loss_tf(KL_pred_tf, obs, done)
                loss_KL.backward()
                self.agent.KL_optimizer.step()

            if self.agent.do_reward:
                Q_ref_pred_tf = self.agent.Q_ref(past_obs, past_action, tf=True)
                loss_Q_ref = self.Q_ref_loss_tf(Q_ref_pred_tf, obs, reward, done)
                loss_Q_ref.backward()
                self.agent.Q_ref_optimizer.step()
                
            if True:
                Q_var_pred_tf = self.agent.Q_var(past_obs, past_action, tf=True)
                KL_pred_tf = self.calc_sum_future_KL(past_obs, past_obs, done, tf=True) - torch.FloatTensor([self.KL(past_obs, done=done)])
                #loss_Q_var = self.Q_ref_loss_tf(Q_var_pred_tf, obs, reward, done) 
                loss_Q_var = self.Q_var_loss_tf(Q_var_pred_tf, KL_pred_tf, obs, reward, done)
                try:
                    loss = KL_pred_tf #loss_Q_var #+ 
                    loss.backward()
                    self.agent.Q_var_optimizer.step()
                except:
                    pass
                
            self.agent.KL_optimizer.zero_grad()
            self.agent.Q_ref_optimizer.zero_grad()
            self.agent.Q_var_optimizer.zero_grad()



    def monte_carlo_update(self, done):
        if done:
            current_time = self.agent.get_time()
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

                if self.agent.isDiscrete:

                    TD_err_ref = self.calc_TD_err_ref(np.sum(liste_reward[time:]), past_obs_or_time, past_action)
                    self.agent.Q_ref[past_obs_or_time, past_action] += self.agent.ALPHA * TD_err_ref

                    TD_err_var = self.calc_TD_err_var(np.sum(liste_reward[time:]),
                                                      np.sum(liste_KL[time:]),
                                                      past_obs,
                                                      past_obs_or_time,
                                                      past_action)
                    self.agent.Q_var[past_obs_or_time, past_action] += self.agent.ALPHA * TD_err_var

                else:
                    pass ## TODO!!


    def run_episode(self):
        self.agent.init_env()
        self.init_trial()
        self.trajectory.append(self.agent.get_observation())

        while True:

            past_time = self.agent.get_time()
            past_obs = self.agent.get_observation()
            past_obs_or_time, past_action, obs_or_time, reward, done = self.agent.step()
            current_time = self.agent.get_time()
            obs = self.agent.get_observation()

            self.action_history.append(past_action)
            self.trajectory.append(obs)

            if True: #not self.final:
                if self.agent.isDiscrete:
                    self.nb_visits[obs] += 1
                    self.obs_score *= 1 - self.OBS_LEAK
                    self.obs_score[obs] += 1
                    self.agent.num_episode += 1
                else:
                    self.mem_obs += [obs]
            if done:
                if self.agent.isDiscrete:
                    self.nb_visits_final[obs] += 1
                    self.obs_score_final *= 1 - self.OBS_LEAK
                    self.obs_score_final[obs] += 1
                else:
                    self.mem_obs_final += [obs]

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
                    V = np.zeros(self.agent.N_obs)
                    for obs in range(self.agent.N_obs):
                        V[obs] = self.agent.softmax_expectation(obs, self.agent.set_Q_obs(obs))
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
               * self.agent.KL(past_obs, past_action)