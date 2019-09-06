import numpy as np
import gym
from environment import Environment
import torch
import torch.nn as nn

class Agent:

    def __init__(self, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, isTime=False, do_reward = True):
        #self.total_reward = 0.0
        self.env = env
        self.init_env()
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.num_episode = 0
        self.do_reward = do_reward
        self.isTime = isTime
        self.isGym = type(env) is not Environment
        if self.isGym:
            N_INPUT = np.prod(env.observation_space.shape) + env.action_space.n
            N_HIDDEN = 50
            self.KL = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_ref = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_var = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
        else:
            if isTime:
                self.Q_ref = np.zeros((self.env.total_steps, self.env.N_act))  # target Q
                self.Q_var = np.zeros((self.env.total_steps, self.env.N_act))  # variational Q
            else:
                self.Q_ref = np.zeros((self.env.N_obs, self.env.N_act))  # target Q
                self.Q_var = np.zeros((self.env.N_obs, self.env.N_act))  # variational Q
            self.KL = np.zeros((self.env.N_obs, self.env.N_act))


    @classmethod
    def timeAgent(cls, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, do_reward=False):
        return cls(env, ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA, isTime=True, do_reward=do_reward)

    def init_env(self):
        self.observation = self.env.reset()
        self.time = 0
        return self.env.reset()

    def get_observation(self):
        return self.observation

    def get_time(self):
        return self.time

    def step(self):
        if self.isTime:
            curr_obs_or_time = self.get_time()
        else:
            curr_obs_or_time = self.get_observation()
        action = self.softmax_choice(curr_obs_or_time)
        new_obs, reward, done, _ = self.env.step(action)
        self.time += 1
        return curr_obs_or_time, action, new_obs, reward, done
        #self.total_reward += reward

    def softmax(self, obs, Q = None):
        if Q is None:
            Q_obs = self.Q_var[obs, :]
        else:
            Q_obs = Q[obs, :]
        act_score = np.zeros(self.env.N_act)
        for a in range(self.env.N_act):
            act_score[a] = np.exp(self.BETA * Q_obs[a])
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
            return np.dot(act_probs, Q)

    def calc_state_probs(self, obs):
        act_probs = self.softmax(obs)
        state_probs = np.zeros(self.env.N_obs)
        for a in range(self.env.N_act):
            if self.env.direction[a] in self.env.next[obs]:
                state_probs[self.env.next[obs][self.env.direction[a]]] += act_probs[a]
            else:
                state_probs[obs] += act_probs[a]
        return state_probs

