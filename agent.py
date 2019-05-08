import numpy as np

class Agent:

    def __init__(self, env, ALPHA=0.1, GAMMA=0.9, BETA=0.5, isTime=False, do_reward = True):
        #self.total_reward = 0.0
        self.env = env
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.num_episode = 0
        self.do_reward = do_reward
        self.isTime = isTime
        if isTime:
            self.Q_ref = np.zeros((self.env.total_steps, self.env.N_act))  # target Q
            self.Q_var = np.zeros((self.env.total_steps, self.env.N_act))  # variational Q
            self.KL_diff = np.zeros((self.env.total_steps, self.env.N_act))
        else:
            self.Q_ref = np.zeros((self.env.N_obs, self.env.N_act))  # target Q
            self.Q_var = np.zeros((self.env.N_obs, self.env.N_act))  # variational Q
            self.KL_diff = np.zeros((self.env.N_obs, self.env.N_act))

    @classmethod
    def timeAgent(cls, env, ALPHA=0.1, GAMMA=0.9, BETA=0.5):
        return cls(env, ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA, isTime = True)

    def init_env(self, total_steps=10):
        self.env.env_initialize()

    def step(self):
        if self.isTime:
            curr_obs = self.env.get_time()
        else:
            curr_obs = self.env.get_observation()
        action = self.softmax_choice(curr_obs)
        new_obs, reward, done = self.env.act(action)
        return curr_obs, action, new_obs, reward, done
        #self.total_reward += reward

    def softmax(self, obs):
        act_score = np.zeros(self.env.N_act)
        for a in range(self.env.N_act):
            act_score[a] = np.exp(self.BETA * self.Q_var[obs, a])
        return act_score / np.sum(act_score)

    def softmax_choice(self, obs):
        act_probs = self.softmax(obs)
        action = np.random.choice(len(act_probs), p=act_probs)
        return action

    def softmax_expectation(self, obs, Q = None):
        act_probs = self.softmax(obs)
        if Q is None:
            return np.dot(act_probs, self.Q_var[obs,:])
        else:
            #act_probs = self.softmax(obs, Q = Q)
            return np.dot(act_probs, Q[obs, :])

    def calc_state_probs(self, obs):
        act_probs = self.softmax(obs)
        state_probs = np.zeros(self.env.N_obs)
        for a in range(self.env.N_act):
            if self.env.direction[a] in self.env.next[obs]:
                state_probs[self.env.next[obs][self.env.direction[a]]] += act_probs[a]
            else:
                state_probs[obs] += act_probs[a]
        return state_probs

class Trainer():

    def __init__(self, agent, OBS_LEAK = 1e-3):
        self.agent = agent
        self.nb_trials = 0
        self.init_trial(update = False)
        self.nb_visits = np.zeros(self.agent.env.N_obs)
        self.obs_score = np.zeros(self.agent.env.N_obs)
        self.nb_visits_final = np.zeros(self.agent.env.N_obs)
        self.obs_score_final = np.zeros(self.agent.env.N_obs)
        self.OBS_LEAK = OBS_LEAK
        self.mem_V = {}

    def init_trial(self, update = True):
        if update:
            self.nb_trials += 1
        self.total_reward = 0
        self.trajectory = []

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

        p = np.zeros(self.agent.env.N_obs)
        p[np.where(self.nb_visits>0)] = 1/np.sum(self.nb_visits>0)
        return p

        # state_probs = np.zeros(self.agent.env.N_obs)
        # for o in range(self.agent.env.N_obs):
        #     state_probs[o] = 7 - o
        # return state_probs / sum(state_probs)

        # ref_probs = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
        # ref_probs[2] += (1 - EPSILON)
        # return ref_probs

    # agent.KL_diff update
    def KL_diff(self, past_obs, a, new_obs, done=False):
        return 0

    def KL_diff_err(self, past_obs, past_action, obs, done = False):
        if done:
            return self.KL_diff(past_obs, past_action, obs, done = done) - self.agent.KL_diff[past_obs, past_action]
        else:
            return self.agent.GAMMA * self.agent.softmax_expectation(obs,
                                                                     Q = self.agent.KL_diff) - \
                   self.agent.KL_diff[past_obs, past_action]

    def TD_err_ref(self, past_obs, past_action, obs, reward, done):
        if done:
            return reward - self.agent.Q_ref[past_obs, past_action]
        else:
            return reward + \
                   self.agent.GAMMA * self.agent.softmax_expectation(obs,
                                                                     Q=self.agent.Q_ref) - \
                   self.agent.Q_ref[past_obs, past_action]

    # For agent.Q_var update
    def KL_diff_loss(self, past_obs, a, new_obs, done=False):
        return 0

    def TD_err_var(self, past_obs, past_action, obs, reward, done):
        if done:
            return reward - self.agent.Q_var[past_obs, past_action] - self.KL_diff_loss(past_obs, past_action, obs)
        else:
            return reward + \
                   self.agent.GAMMA * self.agent.softmax_expectation(obs,
                                                                     Q=self.agent.Q_ref) - \
                   self.agent.Q_var[past_obs, past_action] - \
                   self.KL_diff_loss(past_obs, past_action, obs, done = done)

    def run_episode(self):
        self.agent.init_env()
        self.init_trial()
        self.trajectory.append(self.agent.env.get_observation())
        while True:
            past_obs, past_action, obs, reward, done = self.agent.step()
            self.nb_visits[obs] += 1
            self.obs_score *= 1 - self.OBS_LEAK
            self.obs_score[obs] += self.OBS_LEAK
            self.agent.num_episode += 1
            if done:
                self.nb_visits_final[obs] += 1
                self.obs_score_final *= 1 - self.OBS_LEAK
                self.obs_score_final[obs] += self.OBS_LEAK
            if not self.agent.do_reward:
                reward = 0
            self.total_reward += reward
            self.agent.KL_diff[past_obs, past_action] += self.agent.ALPHA * self.KL_diff_err(past_obs, past_action, obs, done=done)
            self.agent.Q_ref[past_obs, past_action] += self.agent.ALPHA * self.TD_err_ref(past_obs, past_action, obs, reward, done=done)
            self.agent.Q_var[past_obs, past_action] += self.agent.ALPHA * self.TD_err_var(past_obs, past_action, obs, reward, done=done)
            self.trajectory.append(obs)
            V = np.zeros(self.agent.env.N_obs)

            if done:
                for s in range(self.agent.env.N_obs):
                    V[s] = self.agent.softmax_expectation(s)
                self.mem_V[self.nb_trials] = V
                break

class Q_learning_trainer(Trainer):

    def __init__(self, agent):
        super().__init__(agent)

class one_step_variational_trainer(Trainer):

    def __init__(self, agent):
        super().__init__(agent)

    # agent.Q_var update
    def KL_diff_loss(self, past_obs, a, new_obs, done = False):
        pi = self.agent.softmax(past_obs)[a]
        state_probs = self.agent.calc_state_probs(past_obs)
        ref_probs = self.calc_ref_probs(past_obs)
        return (1 - pi) * (np.log(state_probs[new_obs]) + pi / state_probs[new_obs] * 1 - np.log(ref_probs[new_obs]))

class final_variational_trainer(Trainer):

    def __init__(self, agent):
        super().__init__(agent)

    def KL_diff(self, past_obs, a, final_obs, done = False):
        if done:
            state_probs = self.calc_final_state_probs(final_obs)
            ref_probs = self.calc_ref_probs(final_obs, EPSILON=1e-10)
            #print('KL loss diff :', pi * (1 - pi) * (np.log(state_probs[new_obs]) + 1 ))
            #return np.log(state_probs[final_obs]) + 1 - np.log(ref_probs[final_obs])
            return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs])  #+1
        else:
            return self.agent.KL_diff[past_obs, a]

    # agent.Q_var update
    def KL_diff_loss(self, past_obs, a, final_obs, done = False):
        pi = self.agent.softmax(past_obs)[a]
        return (1-pi) * self.KL_diff(past_obs, a, final_obs, done)


