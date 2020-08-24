import gym
import roboschool
from environment import Environment
from agent import Agent, Transition
from trainer import Final_variational_trainer, Q_learning_trainer
import numpy as np
import math

import torch

import time
import os

import json
from easydict import EasyDict
from datetime import date

# Environment

args = EasyDict()
args.ENV_NAME = 'BipedalWalker-v2'

env = gym.make(args.ENV_NAME)

# Trainer

args.monte_carlo=True
args.augmentation = True
args.final = False 
args.OBS_LEAK = 1e-3 
args.N_PART = 1000
args.KNN_prob=True

# Agent

args.isTime=False
args.offPolicy = False
ALPHA_REF = 1e-4
args.BETA = 100 
args.PREC = 1 
args.GAMMA=0.99 ######## !!!!!!!! ########
args.Q_VAR_MULT = 30
args.do_reward = True
args.HIST_HORIZON = 200 * int(1/args.OBS_LEAK)
args.N_HIDDEN = 300
args.optim = 'Adam'
args.ALPHA = ALPHA_REF / args.Q_VAR_MULT / args.PREC 
args.act_renorm = False
args.retain_present = True

# Data path

data_path = "data/{}/{}-{}".format(args.ENV_NAME, str(date.today()), args.ENV_NAME)
if not args.final:
    data_path += '-full'
else:
    data_path += '-final'
if args.do_reward:
    data_path += '-do-reward'
else:
    data_path += '-no-reward'
data_path += '-{}'.format(args.optim)
data_path += '-LEAK-{}'.format(args.OBS_LEAK)
if args.KNN_prob:
    data_path += '-KNN'    
data_path += '-PART-{}'.format(args.N_PART)
data_path += '-HIDDEN-{}'.format(args.N_HIDDEN)
data_path += '-GAMMA-{}'.format(args.GAMMA)
data_path += '-BETA-{}'.format(args.BETA)
data_path += '-PREC-{}'.format(args.PREC)
if args.retain_present:
    data_path += '-retain'
data_path += '-ALPHA-{}'.format(ALPHA_REF)

data_path_npy = data_path+'.npy'
data_path_json = data_path+'.json'
data_path_Q_var = data_path+'Q_var.pt'

if not os.path.isfile(data_path_json):
    with open(data_path_json, 'w') as fp:
        json.dump(args, fp)
        
agent = Agent(env,
          ALPHA=args.ALPHA,
          BETA=args.BETA, 
          GAMMA=args.GAMMA, 
          PREC=args.PREC,
          do_reward=args.do_reward,
          Q_VAR_MULT=args.Q_VAR_MULT,
          isTime=args.isTime,    #!! TimeAgent
          offPolicy=args.offPolicy,
          HIST_HORIZON = args.HIST_HORIZON,
          optim=args.optim,
          N_HIDDEN=args.N_HIDDEN,
          act_renorm=args.act_renorm) 
    
trainer = Final_variational_trainer(agent, 
                                monte_carlo=args.monte_carlo, 
                                augmentation=args.augmentation,
                                final=args.final,
                                OBS_LEAK=args.OBS_LEAK,
                                ref_prob='unif',
                                N_PART=args.N_PART,
                                KNN_prob=args.KNN_prob)

num_steps = 0
step_max = 1e6
if not os.path.isfile(data_path_npy):
    while num_steps < step_max:
        print('***' + str(trainer.nb_trials) + '***')
        print('BETA: ', args.BETA, ', PREC :', args.PREC, ', ALPHA:', ALPHA_REF, ', LEAK:', args.OBS_LEAK)
        trainer.run_episode()
        num_steps += agent.get_time()
        #print("Trajectory: ", trainer.trajectory)
        print("  total reward got: %.4f" % trainer.total_reward)
        print('  mean rtg:', np.mean(trainer.rtg_history))
        print('  total num_steps :' + str(num_steps))
        print("  #time steps : %d" % len(trainer.trajectory))
        if True : #trainer.nb_trials%10 ==0:
    
            transitions = agent.memory.sample(20)
            batch = Transition(*zip(*transitions))
            obs_sample = batch.obs
            act_sample = batch.action

            data_tuple = (trainer.mem_total_reward, 
                          trainer.mem_t_final,
                          trainer.mem_mean_rtg)
            sample_tuple = ( 
                          obs_sample,
                          act_sample
                          )
            data = np.array((data_tuple, sample_tuple))
            np.save(data_path+'.npy', data)
            
            torch.save(agent.Q_var_nn, data_path_Q_var)
