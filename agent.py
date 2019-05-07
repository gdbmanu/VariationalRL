import numpy as np
from TP1_environment import Environment

class Agent:

    def __init__(self, ALPHA=0.1, GAMMA=0.9, BETA = 0.5):
        #self.total_reward = 0.0
        self.env = Environment()
        self.Q = np.zeros((self.env.N_obs, self.env.N_act))
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.num_episode = 0

    def init_env(self):
        self.env = Environment()

    def step(self):
        curr_obs = self.env.get_observation()
        action = self.softmax_choice(curr_obs)
        new_obs, reward, done = self.env.act(action)
        return curr_obs, action, new_obs, reward, done
        #self.total_reward += reward

    def softmax(self, obs, Q = None):
        if Q is None:
            Q = self.Q
        act_score = np.zeros(self.env.N_act)
        for a in range(self.env.N_act):
            act_score[a] = np.exp(self.BETA  * Q[obs, a])
        return act_score / np.sum(act_score)

    def softmax_choice(self, obs):
        act_probs = self.softmax(obs)
        action = np.random.choice(len(act_probs), p=act_probs)
        return action

    def softmax_expectation(self, obs, Q = None):
        act_probs = self.softmax(obs)
        if Q is None:
            return np.dot(act_probs, self.Q[obs,:])
        else:
            #act_probs = self.softmax(obs, Q = Q)
            return np.dot(act_probs, Q[obs, :])

    def calc_state_probs(self, obs):
        act_probs = self.softmax(obs)
        state_probs = np.zeros(self.env.N_obs)
        for a in range(self.env.N_act):
            if Environment.direction[a] in Environment.next[obs]:
                state_probs[Environment.next[obs][Environment.direction[a]]] += act_probs[a]
            else:
                state_probs[obs] += act_probs[a]
        return state_probs

class Q_learning_trainer:

    def __init__(self, agent):
        self.agent = agent
        self.total_reward = 0
        self.trajectory = []

    def init_trainer(self):
        self.total_reward = 0
        self.trajectory = []

    def run_episode(self):
        self.agent.init_env()
        self.init_trainer()
        self.trajectory.append(self.agent.env.get_observation())
        while True:
            past_obs, past_action, obs, reward, done = self.agent.step()
            self.total_reward += reward
            if done:
                TD_err = reward - self.agent.Q[past_obs, past_action]
            else:
                TD_err = reward + self.agent.GAMMA * self.agent.softmax_expectation(obs) - self.agent.Q[past_obs, past_action]
            self.agent.Q[past_obs, past_action] += self.agent.ALPHA * TD_err
            self.trajectory.append(obs)
            if done:
                self.agent.num_episode += 1
                break

class one_step_variational_trainer:

    def __init__(self, agent):
        self.agent = agent
        self.total_reward = 0
        self.trajectory = []
        self.Q_ref = np.zeros((self.agent.env.N_obs, self.agent.env.N_act))
        self.nb_visits = np.zeros(self.agent.env.N_obs)

    def init_trainer(self):
        self.total_reward = 0
        self.trajectory = []

    def calc_ref_probs(self, obs):

        EPSILON = 1e-1
        # state_probs = np.ones(self.agent.env.N_obs) * EPSILON
        # for a in range(self.agent.env.N_act):
        #    if Environment.direction[a] in Environment.next[obs]:
        #        state_probs[Environment.next[obs][Environment.direction[a]]] = 1
        # return state_probs / sum(state_probs)

        # state_probs = np.zeros(self.agent.env.N_obs)
        # for o in range(self.agent.env.N_obs):
        #     state_probs[o] = 7 - o
        # return state_probs / sum(state_probs)

        #
        state_probs = np.ones(self.agent.env.N_obs) * EPSILON/self.agent.env.N_obs
        state_probs[2] += (1-EPSILON)
        return state_probs

    def KL_diff_loss(self, past_obs, a, new_obs):
        pi = self.agent.softmax(past_obs)[a]
        state_probs = self.agent.calc_state_probs(past_obs)
        ref_probs = self.calc_ref_probs(past_obs)
        #print('KL loss diff :', pi * (1 - pi) * (np.log(state_probs[new_obs]) + 1 ))
        return (1 - pi) * (np.log(state_probs[new_obs]) + pi/state_probs[new_obs] - np.log(ref_probs[new_obs]))

    def run_episode(self):
        self.agent.init_env()
        self.init_trainer()
        self.trajectory.append(self.agent.env.get_observation())
        while True:
            past_obs, past_action, obs, reward, done = self.agent.step()
            self.nb_visits[obs] += 1
            reward *= 0
            self.total_reward += reward
            if done:
                TD_err_ref = reward - self.Q_ref[past_obs, past_action]
                TD_err = reward - self.agent.Q[past_obs, past_action]
            else:
                TD_err_ref = reward + self.agent.GAMMA * self.agent.softmax_expectation(obs, Q = self.Q_ref) - self.Q_ref[past_obs, past_action]
                TD_err = reward + self.agent.GAMMA * self.agent.softmax_expectation(obs, Q=self.Q_ref) - self.agent.Q[
                past_obs, past_action]
            explo_diff = self.KL_diff_loss(past_obs, past_action, obs)
            print(TD_err, explo_diff)
            self.Q_ref[past_obs, past_action] += self.agent.ALPHA * TD_err_ref
            self.agent.Q[past_obs, past_action] += self.agent.ALPHA * (TD_err - explo_diff)
            self.trajectory.append(obs)
            if done:
                self.agent.num_episode += 1
                break

class final_variational_trainer:

    def __init__(self, agent):
        self.agent = agent
        self.total_reward = 0
        self.trajectory = []
        self.Q_ref = np.zeros((self.agent.env.N_obs, self.agent.env.N_act))
        self.KL_diff = np.zeros((self.agent.env.N_obs, self.agent.env.N_act))
        #self.model = np.ones(self.agent.env.N_obs) * 1/self.agent.env.N_obs
        self.nb_visits = np.zeros(self.agent.env.N_obs)
        self.obs_score = np.zeros(self.agent.env.N_obs)

    def init_trainer(self):
        self.total_reward = 0
        self.trajectory = []

    def calc_state_probs(self, obs):
        #return self.nb_visits/np.sum(self.nb_visits)
        return self.obs_score/np.sum(self.obs_score)

    def calc_ref_probs(self, obs):
        EPSILON = 1e-15
        p = np.ones(self.agent.env.N_obs) * EPSILON / self.agent.env.N_obs
        p[2] += (1 - EPSILON)
        #p = np.zeros(self.agent.env.N_obs)
        #p[np.where(self.nb_visits>0)] = 1/np.sum(self.nb_visits>0)
        return p

    def final_KL_diff_loss(self, final_obs):
        #pi = self.agent.softmax(past_obs)[a]
        state_probs = self.calc_state_probs(final_obs)
        ref_probs = self.calc_ref_probs(final_obs)
        #print('KL loss diff :', pi * (1 - pi) * (np.log(state_probs[new_obs]) + 1 ))
        #return np.log(state_probs[final_obs]) + 1 - np.log(ref_probs[final_obs])
        return np.log(state_probs[final_obs]) - np.log(ref_probs[final_obs]) #+1

    def run_episode(self):
        self.agent.init_env()
        self.init_trainer()
        self.trajectory.append(self.agent.env.get_observation())
        while True:
            past_obs, past_action, obs, reward, done = self.agent.step()
            reward *= 0
            self.total_reward += reward
            if done:
                EPS = 1e-3
                self.nb_visits[obs] += 1
                self.obs_score *= 1 - EPS
                self.obs_score[obs] += EPS
                TD_err_ref = reward - self.Q_ref[past_obs, past_action]
                TD_err = reward - self.agent.Q[past_obs, past_action]
                TD_err_KL = self.final_KL_diff_loss(obs) - self.KL_diff[past_obs, past_action]
                print(obs, TD_err_KL)
            else:
                TD_err_ref = reward + self.agent.GAMMA * self.agent.softmax_expectation(obs, Q = self.Q_ref) - self.Q_ref[past_obs, past_action]
                TD_err = reward + self.agent.GAMMA * self.agent.softmax_expectation(obs, Q = self.Q_ref) - self.agent.Q[past_obs, past_action]
                #TD_err_KL = self.agent.GAMMA * self.agent.softmax_expectation(obs, Q = self.KL_diff) - self.KL_diff[past_obs, past_action]
                TD_err_KL = self.agent.GAMMA * self.agent.softmax_expectation(obs, Q = self.KL_diff) - self.KL_diff[past_obs, past_action]
                #print(obs, self.agent.softmax_expectation(obs, Q = self.KL_diff), self.KL_diff[past_obs, past_action])
                print(obs, TD_err_KL)
            self.KL_diff[past_obs, past_action] += self.agent.ALPHA * TD_err_KL
            self.Q_ref[past_obs, past_action] += self.agent.ALPHA * TD_err_ref
            #self.agent.Q[past_obs, past_action] += self.agent.ALPHA *  TD_err
            pi = self.agent.softmax(past_obs)[past_action]
            self.agent.Q[past_obs, past_action] += self.agent.ALPHA  * (TD_err - (1-pi) * (self.KL_diff[past_obs, past_action]))
            #self.agent.Q[past_obs, past_action] += self.agent.ALPHA * (TD_err -  TD_err_KL)
            self.trajectory.append(obs)
            if done:
                self.agent.num_episode += 1
                break