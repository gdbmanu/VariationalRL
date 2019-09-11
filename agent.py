import numpy as np
import gym
from environment import Environment
import torch
import torch.nn as nn

class Agent:

    def __init__(self, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, isTime=False, do_reward = True):
        #self.total_reward = 0.0
        self.env = env
        self.isDiscrete = type(env) is Environment or type(env) is gym.spaces.discrete.Discrete
        self.init_env()
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.num_episode = 0
        self.do_reward = do_reward
        self.isTime = isTime
        if not self.isDiscrete:
            N_INPUT = self.N_obs + self.N_act
            N_HIDDEN = 50
            self.KL_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.KL_optimizer = torch.optim.Adam(self.KL_nn.parameters(), lr = self.ALPHA * 10)
            self.Q_ref_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_ref_optimizer = torch.optim.Adam(self.Q_ref_nn.parameters(), lr = self.ALPHA)
            self.Q_var_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_var_optimizer = torch.optim.Adam(self.Q_var_nn.parameters(), lr=self.ALPHA)
        else:
            if self.isTime:
                self.Q_ref_tab = np.zeros((self.env.total_steps, self.N_act))  # target Q
                self.Q_var_tab = np.zeros((self.env.total_steps, self.N_act))  # variational Q
            else:
                self.Q_ref_tab = np.zeros((self.N_obs, self.N_act))  # target Q
                self.Q_var_tab = np.zeros((self.N_obs, self.N_act))  # variational Q
            self.KL_tab = np.zeros((self.N_obs, self.N_act))


    @classmethod
    def timeAgent(cls, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, do_reward=False):
        return cls(env, ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA, isTime=True, do_reward=do_reward)

    def init_env(self):
        self.observation = self.env.reset()
        self.time = 0
        if self.isDiscrete:
            self.N_act = self.env.N_act
            self.N_obs = self.env.N_obs
        else:
            self.N_act = self.env.action_space.n
            self.N_obs = np.prod(self.env.observation_space.shape)
        return self.env.reset()

    def get_observation(self):
        return self.observation

    def get_time(self):
        return self.time

    def one_hot(self, act):
        out = np.zeros(self.N_act)
        out[act] = 1
        return out

    def tf_normalize(self, obs):
        m = (self.env.observation_space.high + self.env.observation_space.low) / 2
        diff = self.env.observation_space.high - self.env.observation_space.low
        return (obs - m) / diff

    def KL(self, obs, act, tf=False):
        if self.isDiscrete:
            return self.KL_tab[obs, act]
        else:
            norm_obs = self.tf_normalize(obs)
            input = np.concatenate((norm_obs, self.one_hot(act)))
            obs_tf = torch.FloatTensor([input])
            if tf:
                return self.KL_nn(obs_tf)
            else:
                return self.KL_nn(obs_tf).data.numpy()[0]

    def Q_ref(self, obs_or_time, act, tf=False):
        if self.isDiscrete:
            return self.Q_ref_tab[obs_or_time, act]
        else:
            norm_obs_or_time = self.tf_normalize(obs_or_time)
            input = np.concatenate((norm_obs_or_time, self.one_hot(act)))
            obs_tf = torch.FloatTensor([input])
            if tf:
                return self.Q_ref_nn(obs_tf)
            else:
                return self.Q_ref_nn(obs_tf).data.numpy()[0]

    def Q_var(self, obs_or_time, act, tf=False):
        if self.isDiscrete:
            return self.Q_var_tab[obs_or_time, act]
        else:
            norm_obs_or_time = self.tf_normalize(obs_or_time)
            input = np.concatenate((norm_obs_or_time, self.one_hot(act)))
            obs_tf = torch.FloatTensor([input])
            if tf:
                return self.Q_var_nn(obs_tf)
            else:
                return self.Q_var_nn(obs_tf).data.numpy()[0]

    def set_Q_obs(self, obs, Q=None):
        # if self.isDiscrete:
        #     if Q is None:
        #         return self.Q_var_tab[obs, :]
        #     else:
        #         return Q[obs, :]
        # else:
        if Q is None:
            Q = self.Q_var
        Q_obs = np.zeros(self.N_act)
        for a in range(self.N_act):
            Q_obs[a] = Q(obs, a)
        return Q_obs

    def softmax(self, obs, Q=None):
        Q_obs = self.set_Q_obs(obs, Q=Q)
        act_score = np.zeros(self.N_act)
        for a in range(self.N_act):
            act_score[a] = np.exp(self.BETA * Q_obs[a])
        return act_score / np.sum(act_score)

    def softmax_choice(self, obs):
        act_probs = self.softmax(obs)
        action = np.random.choice(self.N_act, p=act_probs)
        return action

    def softmax_expectation(self, obs, next_values):
        act_probs = self.softmax(obs)
        #Q_obs = self.set_Q_obs(obs, Q=Q)
        return np.dot(act_probs, next_values)

    def step(self):
        if self.isTime:
            curr_obs_or_time = self.get_time()
        else:
            curr_obs_or_time = self.get_observation()
        action = self.softmax_choice(curr_obs_or_time)
        new_obs, reward, done, _ = self.env.step(action)
        self.observation = new_obs
        self.time += 1
        return curr_obs_or_time, action, new_obs, reward, done
        # self.total_reward += reward


    ## DEPRECATED ??
    def calc_state_probs(self, obs):
        act_probs = self.softmax(obs)
        state_probs = np.zeros(self.N_obs)
        for a in range(self.N_act):
            if self.env.direction[a] in self.env.next[obs]:
                state_probs[self.env.next[obs][self.env.direction[a]]] += act_probs[a]
            else:
                state_probs[obs] += act_probs[a]
        return state_probs

