import numpy as np

class Trainer():

    def __init__(self, agent, OBS_LEAK = 1e-3, EPSILON = 1e-3, ref_prob = 'unif'):
        self.agent = agent
        self.nb_trials = 0
        self.init_trial(update = False)
        self.nb_visits = np.zeros(self.agent.env.N_obs)
        self.obs_score = np.zeros(self.agent.env.N_obs)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        self.OBS_LEAK = OBS_LEAK
        self.EPSILON = EPSILON
        self.mem_V = {}
        self.ref_prob = ref_prob

    def init_trial(self, update = True):
        if update:
            self.nb_trials += 1
        self.total_reward = 0
        self.trajectory = []
        self.action_history = []

    def calc_state_probs(self, obs):
        # return self.nb_visits/np.sum(self.nb_visits)
        return self.obs_score / np.sum(self.obs_score)

    def calc_final_state_probs(self, obs):
        #return self.nb_visits/np.sum(self.nb_visits)
        return self.obs_score_final/np.sum(self.obs_score_final)

    def calc_ref_probs(self, obs, EPSILON = 1e-3):

        # state_probs = np.ones(self.agent.env.N_obs) * EPSILON
        # for a in range(self.agent.env.N_act):
        #    if Environment.direction[a] in Environment.next[obs]:
        #        state_probs[Environment.next[obs][Environment.direction[a]]] = 1
        # return state_probs / sum(state_probs)

        if self.ref_prob == 'unif':
            p = np.zeros(self.agent.env.N_obs)
            p[np.where(self.nb_visits>0)] = 1/np.sum(self.nb_visits>0)
            return p
        else:
            ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
            ref_probs[int(self.ref_prob)] += (1 - EPSILON)
            return ref_probs

        # state_probs = np.zeros(self.agent.env.N_obs)
        # for o in range(self.agent.env.N_obs):
        #     state_probs[o] = 7 - o
        # return state_probs / sum(state_probs)



    # agent.KL_diff update
    def KL(self, past_obs, a, new_obs, done=False):
        return 0

    def KL_err(self, past_obs, past_action, obs, done = False, current_time = None):
        if done:
            return self.KL(past_obs, past_action, obs, done = done) - self.agent.KL[past_obs, past_action]
        else:
            if current_time is None:
                return self.agent.GAMMA * self.agent.softmax_expectation(obs,
                                                                         Q = self.agent.KL[obs,:]) - \
                       self.agent.KL[past_obs, past_action]
            else:
                return self.agent.GAMMA * self.agent.softmax_expectation(current_time,
                                                                         Q=self.agent.KL[obs, :]) - \
                       self.agent.KL[past_obs, past_action]


    def TD_err_ref(self, past_obs, past_action, obs, reward, done=False):
        if done:
            return reward - self.agent.Q_ref[past_obs, past_action]
        else:
            return reward + \
                   self.agent.GAMMA * self.agent.softmax_expectation(obs,
                                                                     Q=self.agent.Q_ref[obs,:]) - \
                   self.agent.Q_ref[past_obs, past_action]

    # For agent.Q_var update
    def KL_diff(self, past_obs, a, new_obs, done=False, past_time = None):
        return 0

    def TD_err_var(self, past_obs, past_action, obs, reward, done=False, past_time=None, current_time=None):
        # if False: #self.agent.Q_ref[past_obs, past_action] == 0:
        #     Q_ref = np.zeros((self.agent.env.N_obs, self.agent.env.N_act))
        #     Q_var = 0
        # else:
        if done:
            if past_time is None:
                return reward - self.agent.Q_var[past_obs, past_action] - self.KL_diff(past_obs, past_action, obs,
                                                                                       past_time = past_time)
            else:
                return reward - self.agent.Q_var[past_time, past_action] - self.KL_diff(past_obs, past_action, obs,
                                                                        past_time=past_time)
        else:
            if past_time is None:
                Q_ref = self.agent.Q_ref[obs, :]
                Q_var = self.agent.Q_var[past_obs, past_action]
                obs_or_time = obs
            else:
                Q_ref = self.agent.Q_ref[current_time, :]
                Q_var = self.agent.Q_var[past_time, past_action]
                obs_or_time = current_time
            return reward + \
                   self.agent.GAMMA * self.agent.softmax_expectation(obs_or_time,
                                                                     Q=Q_ref) - \
                   Q_var - \
                   self.KL_diff(past_obs, past_action, obs, done = done, past_time = past_time)

    def run_episode(self):
        self.agent.init_env()
        self.init_trial()
        self.trajectory.append(self.agent.env.get_observation())
        while True:
            past_time = self.agent.env.get_time()
            if self.agent.isTime:
                mem_obs = self.agent.env.get_observation()
            past_obs, past_action, obs, reward, done = self.agent.step()
            current_time = self.agent.env.get_time()
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
            if not self.agent.do_reward:
                reward = 0
            self.total_reward += reward
            if self.agent.isTime:
                past_obs = mem_obs
                self.agent.KL[past_obs, past_action] += self.agent.ALPHA * self.KL_err(past_obs, past_action, obs,
                                                                                       done=done, current_time=current_time)
                self.agent.Q_ref[past_time, past_action] += self.agent.ALPHA * self.TD_err_ref(past_time, past_action, current_time, reward, done=done)
                self.agent.Q_var[past_time, past_action] += self.agent.ALPHA * self.TD_err_var(past_obs, past_action, obs,
                                                                                               reward, done=done,
                                                                                               past_time=past_time, current_time=current_time)
            else:
                self.agent.KL[past_obs, past_action] += self.agent.ALPHA * self.KL_err(past_obs, past_action, obs,
                                                                                       done=done)
                self.agent.Q_ref[past_obs, past_action] += self.agent.ALPHA * self.TD_err_ref(past_obs, past_action,
                                                                                              obs, reward, done=done)
                self.agent.Q_var[past_obs, past_action] += self.agent.ALPHA * self.TD_err_var(past_obs, past_action,
                                                                                              obs, reward, done=done)
            if done:
                if self.nb_trials % 100 == 0 and not self.agent.isTime:
                    V = np.zeros(self.agent.env.N_obs)
                    for s in range(self.agent.env.N_obs):
                        V[s] = self.agent.softmax_expectation(s)
                    self.mem_V[self.nb_trials] = V
                break

class Q_learning_trainer(Trainer):

    def __init__(self, agent, EPSILON = 1e-3, OBS_LEAK = 1e-3, ref_prob = 'unif'):
        super().__init__(agent, EPSILON = EPSILON, OBS_LEAK = OBS_LEAK, ref_prob = ref_prob)

class One_step_variational_trainer(Trainer):

    def __init__(self, agent, EPSILON = 1e-3, OBS_LEAK = 1e-3, ref_prob = 'unif'):
        super().__init__(agent, EPSILON = EPSILON, OBS_LEAK = OBS_LEAK, ref_prob = ref_prob)

    # agent.Q_var update
    def KL_diff(self, past_obs, a, new_obs, done = False, past_time = None):
        pi = self.agent.softmax(past_obs)[a]
        state_probs = self.agent.calc_state_probs(past_obs)
        ref_probs = self.calc_ref_probs(past_obs, EPSILON=self.EPSILON)
        return (1 - pi) * (np.log(state_probs[new_obs]) + pi / state_probs[new_obs] * 1 - np.log(ref_probs[new_obs]))

class Final_variational_trainer(Trainer):

    def __init__(self, agent, EPSILON = 1e-3, OBS_LEAK = 1e-3, ref_prob = 'unif'):
        super().__init__(agent, EPSILON = EPSILON, OBS_LEAK = OBS_LEAK, ref_prob = ref_prob)

    def KL(self, past_obs, a, final_obs, done = False):
        if done:
            state_probs = self.calc_final_state_probs(final_obs)
            ref_probs = self.calc_ref_probs(final_obs, EPSILON=self.EPSILON)
            return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])  #+1
        else:
            #state_probs = self.calc_state_probs(final_obs)
            #ref_probs = self.calc_ref_probs(final_obs, EPSILON=1e-10)
            #return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])
            return self.agent.KL[past_obs, a]

    # agent.Q_var update
    def KL_diff(self, past_obs, a, final_obs, done = False, past_time = None):
        if past_time is None:
            pi = self.agent.softmax(past_obs)[a]
        else:
            pi = self.agent.softmax(past_time)[a]
        return (1-pi) * self.KL(past_obs, a, final_obs, done)
