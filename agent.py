import numpy as np
import gym
from environment import Environment
import torch
import torch.nn as nn

class Agent:

    def __init__(self, env, ALPHA=0.01, GAMMA=0.9, BETA = 1, PREC=1, isTime=False, do_reward = True,
                 Q_VAR_MULT=30, offPolicy=False):
        #self.total_reward = 0.0
        self.env = env
        self.isDiscrete = type(env) is Environment or type(env) is gym.spaces.discrete.Discrete
        self.continuousAction = type(env) is not Environment and type(env.action_space) is gym.spaces.box.Box
        self.init_env()
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BETA = BETA
        self.PREC = PREC
        self.num_episode = 0
        self.do_reward = do_reward
        self.isTime = isTime
        self.offPolicy = offPolicy
        if not self.isDiscrete:
            N_INPUT = self.N_obs + self.N_act
            N_HIDDEN = 50
            
            self.high = self.env.observation_space.high
            indices_high = np.where(self.high > 1e18)
            self.high[indices_high] = 3
            self.low = self.env.observation_space.low
            indices_low = np.where(self.low < -1e18)
            self.low[indices_low] = -3
            
            if self.continuousAction:
                self.act_high = self.env.action_space.high
                self.act_low = self.env.action_space.low
            
            self.Q_KL_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, N_HIDDEN, bias=True),
                nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_KL_optimizer = torch.optim.Adam(self.Q_KL_nn.parameters(), lr = self.ALPHA) #!! *30 ?? TODO : à vérifier
            self.Q_ref_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                #nn.Linear(N_HIDDEN, N_HIDDEN, bias=True),
                #nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_ref_optimizer = torch.optim.Adam(self.Q_ref_nn.parameters(), lr = self.ALPHA)
            self.Q_var_nn = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN, bias=True),
                nn.ReLU(),
                #nn.Linear(N_HIDDEN, N_HIDDEN, bias=True),
                #nn.ReLU(),
                nn.Linear(N_HIDDEN, 1, bias=True)
            )
            self.Q_var_optimizer = torch.optim.Adam(self.Q_var_nn.parameters(), lr=self.ALPHA * Q_VAR_MULT )
        else:
            if self.isTime:
                self.Q_ref_tab = 1e-1 * np.random.randn(self.env.total_steps, self.N_act)  # target Q
                self.Q_var_tab = 1e-1 * np.random.randn(self.env.total_steps, self.N_act)  # variational Q
                self.Q_KL_tab = 1e-1 * np.random.randn(self.env.total_steps, self.N_act)
            else:
                self.Q_ref_tab = 1e-1 * np.random.randn(self.N_obs, self.N_act)  # target Q
                self.Q_var_tab = 1e-1 * np.random.randn(self.N_obs, self.N_act)  # variational Q
                self.Q_KL_tab = 1e-1 * np.random.randn(self.N_obs, self.N_act)

    @classmethod
    def timeAgent(cls, env, ALPHA=0.1, GAMMA=0.9, BETA = 1, PREC=1, do_reward=False):
        return cls(env, ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA, PREC=PREC, isTime=True, do_reward=do_reward)

    def init_env(self):
        self.observation = self.env.reset()
        self.time = 0
        if self.isDiscrete:
            self.N_act = self.env.N_act
            self.N_obs = self.env.N_obs
        else:
            if self.continuousAction:
                self.N_act = np.prod(self.env.action_space.shape)
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

    def tf_normalize(self, obs, show=False):
        m = (self.high + self.low) / 2
        diff = self.high - self.low
        if show:
            print("obs", obs)
            #print("m", m)
            #print("diff", diff)
            print("normalized", (obs - m) / diff)
        return (obs - m) / diff
    
    def tf_cat(self, obs, act):
        obs = np.array(obs)
        act = np.array(act)
        if self.continuousAction:
            if act.ndim > 1:
                return np.concatenate((obs, act), 1)
            else:
                return np.concatenate((obs, act))
        else:
            if act.ndim > 1:
                return np.concatenate((obs, self.one_hot(act)), 1)
            else:
                return np.concatenate((obs, act))

    def Q_KL(self, obs, act, tf=False):
        if self.isDiscrete:
            return self.Q_KL_tab[obs, act]
        else:
            norm_obs = self.tf_normalize(obs)
            input = self.tf_cat(norm_obs, act)
            input_tf = torch.FloatTensor(input)
            if tf:
                return self.Q_KL_nn(input_tf)
            else:
                with torch.no_grad():
                    return self.Q_KL_nn(input_tf).data.numpy().squeeze()  #[0]

    def Q_ref(self, obs_or_time, act, tf=False):
        if not self.do_reward:
            return 0
        else:
            if self.isDiscrete:
                return self.Q_ref_tab[obs_or_time, act]
            else:
                norm_obs_or_time = self.tf_normalize(obs_or_time, show=False)
                input = self.tf_cat(norm_obs_or_time, act)
                input_tf = torch.FloatTensor(input)
                if tf:
                    #print('input_tf.shape', input_tf.shape)
                    return self.Q_ref_nn(input_tf)
                else:
                    with torch.no_grad():
                        return self.Q_ref_nn(input_tf).data.numpy().squeeze() #[0]

    def Q_var(self, obs_or_time, act, tf=False):
        if self.isDiscrete:
            return self.Q_var_tab[obs_or_time, act]
        else:
            norm_obs_or_time = self.tf_normalize(obs_or_time)
            input = self.tf_cat(norm_obs_or_time, act)
            input_tf = torch.FloatTensor(input)
            if tf:
                return self.Q_var_nn(input_tf)
            else:
                with torch.no_grad():
                    return self.Q_var_nn(input_tf).data.numpy().squeeze() #[0]

    def set_Q_obs(self, obs, Q=None, tf=False, actions_set=None):
        if Q is None:
            Q = self.Q_var
        if not tf:
            if actions_set is None:
                Q_obs = np.zeros(self.N_act)
            else:
                Q_obs = np.zeros(len(actions_set))
        if actions_set is None:
            for a in range(self.N_act):
                if tf:
                    if a == 0:
                        Q_obs = Q(obs, a, tf=tf)
                    else:
                        Q_obs = torch.cat((Q_obs, Q(obs, a, tf=tf)), 0)
                else:
                    Q_obs[a] = Q(obs, a, tf=tf)
        else:
            for indice_a, a in enumerate(actions_set):
                if tf:
                    if indice_a == 0:
                        Q_obs = Q(obs, a, tf=tf)
                    else:
                        Q_obs = torch.cat((Q_obs, Q(obs, a, tf=tf)), 0)
                else:
                    Q_obs[indice_a] = Q(obs, a, tf=tf)
        return Q_obs
        

    def logSoftmax(self, obs, act, Q=None, tf=False, actions_set=None):
        Q_obs = self.set_Q_obs(obs, Q=Q, tf=tf, actions_set=actions_set)
        if actions_set is None:
            N_act = self.N_act
            index_act = act
        else:
            N_act = len(actions_set)
            #print('actions_set', actions_set)
            index_act = np.where(actions_set == act)[0][0] #actions_set.index(act)
        if tf:
            logp = torch.nn.LogSoftmax(dim=0)(self.BETA * Q_obs)
            return logp.view(N_act)[index_act]
        else:
            act_score = np.zeros(N_act)
            m_Q = np.mean(Q_obs)
            max_Q_score = np.max(self.BETA *(Q_obs - m_Q))
            for a in range(N_act):
                Q_score = max(max_Q_score - 30, self.BETA *(Q_obs[a] - m_Q))
                act_score[a] = np.exp(Q_score)
            return self.BETA * (Q_obs[index_act] - m_Q) - np.log(np.sum(act_score))
        
    def softmax(self, obs, Q=None, tf=False, actions_set=None):
        Q_obs = self.set_Q_obs(obs, Q=Q, tf=tf, actions_set=actions_set)
        if actions_set is None:
            N_act = self.N_act
        else:
            N_act = len(actions_set)
        if tf:
            p = torch.nn.Softmax(dim=0)(self.BETA * Q_obs)
            return p.view(N_act)
        else:
            act_score = np.zeros(N_act)
            #m_Q = np.mean(Q_obs)
            max_Q_score = np.max(self.BETA *(Q_obs))
            for a in range(N_act):
                Q_score = max(max_Q_score - 30, self.BETA * Q_obs[a]) - (max_Q_score - 15)
                act_score[a] = np.exp(Q_score)
            return act_score / np.sum(act_score)

    def greedy_act_max(self, Q):
        return np.argmax(Q)

    def epsilon_greedy(self, obs, Q=None, tf=False, actions_set=None, EPS = 0.1):
        Q_obs = self.set_Q_obs(obs, Q=Q, tf=tf, actions_set=actions_set)
        act_max = self.greedy_act_max(Q_obs)
        if actions_set is None:
            N_act = self.N_act
        else:
            N_act = len(actions_set)
        act_probs = np.zeros(N_act)
        for act in range(N_act):
            if act == act_max:
                act_probs[act] = (1 - EPS) + EPS / N_act
            else:
                act_probs[act] = EPS / N_act
        return act_probs

    def softmax_choice(self, obs, actions_set=None):
        act_probs = self.softmax(obs, actions_set=actions_set)
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

    def softmax_expectation(self, obs, next_values, tf=False, actions_set=None):
        act_probs = self.softmax(obs, tf=tf, actions_set=actions_set)
        #Q_obs = self.set_Q_obs(obs, Q=Q)
        if actions_set is None:
            N_act = self.N_act
        else:
            N_act = len(actions_set)
        if tf:
            return torch.dot(act_probs, next_values.view(N_act))
        else:
            return np.dot(act_probs, next_values)

    def step(self, actions_set=None):
        if self.isTime:
            curr_obs_or_time = self.get_time()
        else:
            curr_obs_or_time = self.get_observation()
        if self.offPolicy:
            action = self.epsilon_greedy_choice(curr_obs_or_time, actions_set=actions_set)
        else:
            action = self.softmax_choice(curr_obs_or_time, actions_set=actions_set)
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

