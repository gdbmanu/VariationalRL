import numpy as np
import random
import gym
from environment import Environment


class Agent:

    def __init__(self, env, ALPHA=0.01, GAMMA=0.9, BETA = 1, PREC=1, isTime=False, do_reward = True,
                 Q_VAR_MULT=30, offPolicy=False, HIST_HORIZON=10000, act_renorm=True):
        self.env = env
        self.init_env()
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.PREC = PREC
        self.num_episode = 0
        self.do_reward = do_reward
        self.isTime = isTime
        self.offPolicy = offPolicy
        self.Q_VAR_MULT = Q_VAR_MULT
        self.HIST_HORIZON = HIST_HORIZON
        self.act_renorm=act_renorm
        if self.isTime:
            self.Q_ref_tab = 1e-6 * np.random.uniform(size=(self.env.total_steps, self.N_act))  # target Q
            self.Q_var_tab = 1e-6 * np.random.uniform(size=(self.env.total_steps, self.N_act))  # variational Q
            self.Q_KL_tab = 1e-6 * np.random.uniform(size=(self.env.total_steps, self.N_act))
        else:
            self.Q_ref_tab = 1e-6 * np.random.uniform(size=(self.N_obs, self.N_act))  # target Q
            self.Q_var_tab = 1e-6 * np.random.uniform(size=(self.N_obs, self.N_act))  # variational Q
            self.Q_KL_tab = 1e-6 * np.random.uniform(size=(self.N_obs, self.N_act))

    @classmethod
    def timeAgent(cls, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, PREC=1, do_reward=False):
        return cls(env, ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA, PREC=PREC, isTime=True, do_reward=do_reward)

    def init_env(self):
        self.observation = self.env.reset()
        self.time = 0
        self.N_act = self.env.N_act
        self.N_obs = self.env.N_obs
        return self.env.reset()

    def get_observation(self):
        return self.observation

    def get_time(self):
        return self.time
        
    def one_hot(self, act):
        n_dim = act.ndim
        if n_dim == 0:
            out = np.zeros(act.shape[0]) #self.N_act)
            out[act] = 1
        else:
            out = np.zeros(act.shape) #(len(act), self.N_act))
            for i in range(act.shape[0]): #len(act)):
                out[i, act[i]] = 1
        return out

    def Q_KL(self, obs, act):
        return self.Q_KL_tab[obs, act]


    def Q_ref(self, obs_or_time, act):
        if not self.do_reward:
            return 0
        else:
            return self.Q_ref_tab[obs_or_time, act]


    def Q_var(self, obs_or_time, act):
        return self.Q_var_tab[obs_or_time, act]


    def set_Q_obs(self, obs, Q=None, actions_set=None):
        if Q is None:
            Q = self.Q_var
        if actions_set is None:
            actions_set = range(self.N_act)
        Q_obs = Q(obs, actions_set)
        return Q_obs
        

    def logSoftmax(self, obs, act, Q=None, actions_set=None):
        Q_obs = self.set_Q_obs(obs, Q=Q, actions_set=actions_set)
        if actions_set is None:
            N_act = self.N_act
            index_act = act
        else:
            N_act = len(actions_set)
            #print('actions_set', actions_set)
            index_act = np.where(actions_set == act)[0][0] #actions_set.index(act)

        act_score = np.zeros(N_act)
        m_Q = np.mean(Q_obs)
        max_Q_score = np.max(self.BETA *(Q_obs - m_Q))
        for a in range(N_act):
            Q_score = max(max_Q_score - 30, self.BETA *(Q_obs[a] - m_Q))
            act_score[a] = np.exp(Q_score)
        return self.BETA * (Q_obs[index_act] - m_Q) - np.log(np.sum(act_score))
        
    def softmax(self, obs, Q=None, actions_set=None):
        Q_obs = self.set_Q_obs(obs, Q=Q, actions_set=actions_set)
        if actions_set is None:
            N_act = self.N_act
        else:
            N_act = len(actions_set)

        act_score = np.zeros(N_act)
        #m_Q = np.mean(Q_obs)
        max_Q_score = np.max(self.BETA *(Q_obs))
        for a in range(N_act):
            Q_score = max(max_Q_score - 30, self.BETA * Q_obs[a]) - (max_Q_score - 15)
            act_score[a] = np.exp(Q_score)
        return act_score / np.sum(act_score)

    def greedy_act_max(self, Q):
        if np.abs(np.max(Q)) > 1e-6:
            return np.argmax(Q)
        else:
            return None
       

    def epsilon_greedy(self, obs, Q=None, actions_set=None, EPS = 0.1):
        Q_obs = self.set_Q_obs(obs, Q=Q, actions_set=actions_set)
        act_max = self.greedy_act_max(Q_obs)
        if actions_set is None:
            N_act = self.N_act
        else:
            N_act = len(actions_set)
        act_probs = np.zeros(N_act)
        for act in range(N_act):
            if act_max is not None:
                if act == act_max:
                    act_probs[act] = (1 - EPS) + EPS / N_act
                else:
                    act_probs[act] = EPS / N_act
            else:
                act_probs[act] = 1 / N_act
        return act_probs

    def softmax_choice(self, obs, Q=None, actions_set=None):
        act_probs = self.softmax(obs, Q=Q, actions_set=actions_set)
        if actions_set is None:
            action = np.random.choice(self.N_act, p=act_probs)
        else:
            indice_a = np.random.choice(len(actions_set), p=act_probs)
            action = actions_set[indice_a]
        return action

    def epsilon_greedy_choice(self, obs, actions_set=None):
        act_probs = self.epsilon_greedy(obs, actions_set=actions_set)
        if actions_set is None:
            action = np.random.choice(self.N_act, p=act_probs)
        else:
            indice_a = np.random.choice(len(actions_set), p=act_probs)
            action = actions_set[indice_a]
        return action

    def softmax_expectation(self, obs, next_values,  actions_set=None):
        act_probs = self.softmax(obs,  actions_set=actions_set)
        return np.dot(act_probs, next_values)

    def step(self, actions_set=None, test=False):
        if self.isTime:
            curr_obs_or_time = self.get_time()
        else:
            curr_obs_or_time = self.get_observation()
        if self.offPolicy:
            action = self.epsilon_greedy_choice(curr_obs_or_time, actions_set=actions_set)
        else:
            if not test:
                action = self.softmax_choice(curr_obs_or_time, Q=self.Q_var, actions_set=actions_set)
            else:
                action = self.softmax_choice(curr_obs_or_time, Q=self.Q_ref, actions_set=actions_set)
        new_obs, reward, done, _ = self.env.step(action)
        self.observation = new_obs
        self.time += 1
        return curr_obs_or_time, action, new_obs, reward, done


