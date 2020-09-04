import numpy as np
import random
import gym
from environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


OnlineTransition = namedtuple('OnlineTransition',
                        ('past_obs', 'past_action', 'reward', 'next_obs', 'done'))

Transition = namedtuple('Transition',
                        ('obs', 'action', 'sum_future_KL', 'sum_future_rewards', 'R_tilde'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

    

class Net(nn.Module):
    
    def __init__(self, N_INPUT, N_HIDDEN, act_renorm=False):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(N_INPUT, N_HIDDEN)
        self.fc_out = nn.Linear(N_HIDDEN, 1)
        self.act_renorm = act_renorm   
        
    #def tf_cat(self, obs, act):
    #    obs = np.array(obs) 
    #    act = np.array(act)
    #    if self.act_renorm:
    #        obs = obs / obs.shape[1] #eself.env.observation_space.shape[0]
    #        act = act / act.shape[1] #self.env.action_space.shape[0]
    #         
    #    # TEST-NORM act *= self.env.observation_space.shape[0] / self.env.action_space.shape[0]
    #    if act.ndim > 1:
    #        return np.concatenate((obs, act), 1)
    #    else:
    #        return np.concatenate((obs, act))
        
        
    def forward(self, obs, act):
        x = torch.cat((obs, act), 1) #self.tf_cat(obs, act)
        x = F.relu(self.fc_in(x))
        return self.fc_out(x)
        
    
class V_net(nn.Module):
    
    def __init__(self, N_obs, N_act, N_HIDDEN):
        super(V_net, self).__init__()
        self.fc_obs = nn.Linear(N_obs, N_HIDDEN)
        self.fc_act = nn.Linear(N_act, N_HIDDEN)
        self.fc_obs_hid = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.fc_act_hid = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.fc_out = nn.Linear(N_HIDDEN, 1)
        
    def forward(self, obs, act):  
        x_obs = F.relu(self.fc_obs(obs))
        h_obs_hid = self.fc_obs_hid(x_obs)
        x_act = F.relu(self.fc_act(act))
        h_act_hid = self.fc_act_hid(x_act)
        x_hid = F.relu(h_obs_hid + h_act_hid)
        return self.fc_out(x_hid)

class Agent:

    def __init__(self, env, ALPHA=0.01, GAMMA=0.9, BETA = 1, PREC=1, isTime=False, do_reward = True,
                 Q_VAR_MULT=30, offPolicy=False, optim='SGD', HIST_HORIZON=10000, N_HIDDEN=50, act_renorm=True,
                 do_V_net = False):
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
        self.Q_VAR_MULT = Q_VAR_MULT
        self.HIST_HORIZON = HIST_HORIZON
        self.act_renorm=act_renorm
        if not self.isDiscrete:
            self.memory = ReplayMemory(HIST_HORIZON)
            
            N_INPUT = self.N_obs + self.N_act
            self.N_HIDDEN = N_HIDDEN
            
            self.high = self.env.observation_space.high
            indices_high = np.where(self.high > 1e18)
            self.high[indices_high] = 1
            self.low = self.env.observation_space.low
            indices_low = np.where(self.low < -1e18)
            self.low[indices_low] = -1
            
            if self.continuousAction:
                self.act_high = self.env.action_space.high
                self.act_low = self.env.action_space.low
            
              
            #self.Q_KL_nn = nn.Sequential(
            #    nn.Linear(N_INPUT, self.N_HIDDEN, bias=True),
            #    nn.ReLU(),
            #    #nn.Linear(self.N_HIDDEN, self.N_HIDDEN, bias=True),
            #    #nn.ReLU(),
            #    nn.Linear(self.N_HIDDEN, 1, bias=True)
            #)
            if not do_V_net:
                self.Q_KL_nn = Net(N_INPUT, self.N_HIDDEN, act_renorm=self.act_renorm)
            else:
                self.Q_KL_nn = V_net(self.N_obs, self.N_act, self.N_HIDDEN)  
            if optim == 'Adam':
                self.Q_KL_optimizer = torch.optim.Adam(self.Q_KL_nn.parameters(), lr = self.ALPHA) #!! *30 ?? TODO : à vérifier
            else:
                self.Q_KL_optimizer = torch.optim.SGD(self.Q_KL_nn.parameters(), lr = self.ALPHA)
            
             
            #self.Q_ref_nn = nn.Sequential(
            #    nn.Linear(N_INPUT, self.N_HIDDEN, bias=True),
            #    nn.ReLU(),
            #    #nn.Linear(self.N_HIDDEN, self.N_HIDDEN, bias=True),
            #    #nn.ReLU(),
            #    nn.Linear(self.N_HIDDEN, 1, bias=True)
            #)
            if not do_V_net:
                self.Q_ref_nn = Net(N_INPUT, self.N_HIDDEN, act_renorm=self.act_renorm)
            else:
                self.Q_ref_nn = V_net(self.N_obs, self.N_act, self.N_HIDDEN)   
            for d in self.Q_ref_nn.fc_out.parameters():
                d.data *= 1/self.BETA
            if optim == 'Adam':
                self.Q_ref_optimizer = torch.optim.Adam(self.Q_ref_nn.parameters(), lr = self.ALPHA)
                                                                                                     # TODO: à tester 
            else:
                self.Q_ref_optimizer = torch.optim.SGD(self.Q_ref_nn.parameters(), lr = self.ALPHA)
            
            
            #self.Q_var_nn = nn.Sequential(
            #    nn.Linear(N_INPUT, self.N_HIDDEN, bias=True),
            #    nn.ReLU(),
            #    #nn.Linear(self.N_HIDDEN, self.N_HIDDEN, bias=True),
            #    #nn.ReLU(),
            #    nn.Linear(self.N_HIDDEN, 1, bias=True)
            #)
            if not do_V_net:
                self.Q_var_nn = Net(N_INPUT, self.N_HIDDEN, act_renorm=self.act_renorm)
            else:
                self.Q_var_nn = V_net(self.N_obs, self.N_act, self.N_HIDDEN)   
            for d in self.Q_var_nn.fc_out.parameters():
                d.data *= 1/self.BETA
            if optim == 'Adam':
                self.Q_var_optimizer = torch.optim.Adam(self.Q_var_nn.parameters(), lr=self.ALPHA * Q_VAR_MULT )
            else:
                self.Q_var_optimizer = torch.optim.SGD(self.Q_var_nn.parameters(), lr=self.ALPHA * Q_VAR_MULT )
        else:
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

    def tf_normalize(self, obs, show=False):
        return obs
        '''m = (self.high + self.low) / 2
        diff = self.high - self.low
        if show:
            print("obs", obs)
            #print("m", m)
            #print("diff", diff)
            print("normalized", (obs - m) / diff)
        return (obs - m) / diff'''    
        
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
    
    def get_tf_input(self, obs, act):
        norm_obs = self.tf_normalize(obs)
        if not self.continuousAction:
            act = self.one_hot(act)
        if isinstance(act, list):
            obs_tf = torch.FloatTensor([norm_obs] * len(act))
            act_tf = torch.FloatTensor(act)
            #inputs = []
            #for a in act:
            #    cat_input = self.tf_cat(norm_obs_or_time, a)
            #    inputs.append([cat_input])                   
        else:
            obs_tf = torch.FloatTensor(norm_obs)
            act_tf = torch.FloatTensor(act)
        if obs_tf.dim()==1:
            obs_tf = obs_tf.unsqueeze(0)
            act_tf = act_tf.unsqueeze(0)
        return obs_tf, act_tf
        #inputs = self.tf_cat(norm_obs_or_time, act)
        #inputs_tf = torch.FloatTensor(inputs)

    def Q_KL(self, obs, act, tf=False, act_renorm=True):
        if self.isDiscrete:
            return self.Q_KL_tab[obs, act]
        else:
            #norm_obs = self.tf_normalize(obs)
            #print(norm_obs, self.one_hot(act))
            #input = self.tf_cat(norm_obs, act)
            #obs_tf = torch.FloatTensor(obs)
            #act_tf = torch.FloatTensor(act)
            obs_tf, act_tf = self.get_tf_input(obs, act)
            #print(input, self.one_hot(act))
            #input_tf = torch.FloatTensor(input)
            if tf:
                return self.Q_KL_nn(obs_tf, act_tf)
            else:
                with torch.no_grad():
                    return self.Q_KL_nn(obs_tf, act_tf).data.numpy().squeeze()  #[0]

    def Q_ref(self, obs_or_time, act, tf=False):
        if not self.do_reward:
            return 0
        else:
            if self.isDiscrete:
                return self.Q_ref_tab[obs_or_time, act]
            else:
                obs_tf, act_tf = self.get_tf_input(obs_or_time, act)
                if tf:
                    #print('input_tf.shape', input_tf.shape)
                    return self.Q_ref_nn(obs_tf, act_tf) #(inputs_tf)
                else:
                    with torch.no_grad():
                        return self.Q_ref_nn(obs_tf, act_tf).data.numpy().squeeze() #[0]

    def Q_var(self, obs_or_time, act, tf=False):
        if self.isDiscrete:
            return self.Q_var_tab[obs_or_time, act]
        else:
            obs_tf, act_tf = self.get_tf_input(obs_or_time, act)
            #norm_obs_or_time = self.tf_normalize(obs_or_time)
            #if isinstance(act, list):
            #    inputs = []
            #    for a in act:
            #        cat_input = self.tf_cat(norm_obs_or_time, a)
            #        inputs.append([cat_input])                   
            #else:
            #    inputs = self.tf_cat(norm_obs_or_time, act)
            #inputs_tf = torch.FloatTensor(inputs)
            if tf:
                return self.Q_var_nn(obs_tf, act_tf) #(inputs_tf)
            else:
                with torch.no_grad():   
                    return self.Q_var_nn(obs_tf, act_tf).data.numpy().squeeze() #[0]

    def set_Q_obs(self, obs, Q=None, tf=False, actions_set=None):
        if Q is None:
            Q = self.Q_var
        '''if not tf:
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
                    Q_obs[indice_a] = Q(obs, a, tf=tf)'''
        if actions_set is None:
            actions_set = range(self.N_act)
        Q_obs = Q(obs, actions_set, tf=tf)
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
        if np.abs(np.max(Q)) > 1e-6:
            return np.argmax(Q)
        else:
            return None
       

    def epsilon_greedy(self, obs, Q=None, tf=False, actions_set=None, EPS = 0.1):
        Q_obs = self.set_Q_obs(obs, Q=Q, tf=tf, actions_set=actions_set)
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

