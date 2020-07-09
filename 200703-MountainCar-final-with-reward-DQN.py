import gym
from agent import Agent
from trainer import Final_variational_trainer, Q_learning_trainer
import numpy as np

ENV_NAME = 'MountainCar-v0' #'CartPole-v1' #'FrozenLake-v0' #
env = gym.make(ENV_NAME)

BETA = 50
GAMMA=1
OBS_LEAK = 1e-6 #1e-3
PREC = 2e-4 #0.03 # LAMBDA # regularizer
Q_VAR_MULT = 10
ALPHA = 1e-3 / Q_VAR_MULT #3e-3
augmentation = True
do_reward = True
final = True

N = 1000

offPolicy = False
monte_carlo=True
isTime=False

import time
import os

data_path = '200703-MountainCar-final-with-reward-DQN.npy'
BETA_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
PREC_range = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

if not os.path.isfile(data_path):
    mem_obs_final = {}
    mem_total_reward = {}

    tic = time.clock()
    for BETA in BETA_range:
        mem_obs_final[BETA] = {}
        mem_total_reward[BETA] = {}

        for PREC in PREC_range:
            print("BETA=", BETA, ", PREC=", PREC)
            toc = time.clock()
            print("Elapsed time:", toc - tic)

            agent = Agent(env,
                          ALPHA=ALPHA,
                          GAMMA=GAMMA,
                          BETA=BETA,
                          PREC=PREC,
                          do_reward=do_reward,
                          Q_VAR_MULT=Q_VAR_MULT,
                          isTime=isTime,  # !! TimeAgent
                          offPolicy=offPolicy)
            trainer = Final_variational_trainer(agent,
                                                monte_carlo=monte_carlo,
                                                augmentation=augmentation,
                                                final=final,
                                                OBS_LEAK=OBS_LEAK,
                                                ref_prob='unif',
                                                HIST_HORIZON=200 * int(1 / OBS_LEAK))
            for i in range(N):
                # print(i)
                trainer.run_episode()
                if (i + 1) % 1000 == 0:
                    # plt.figure(figsize = (4, 4))
                    # plt.plot(agent.KL.flatten())
                    print(trainer.nb_trials)
                    # print("Trajectory: ", trainer.trajectory)
                    print("Total reward got: %.4f" % trainer.total_reward)

            mem_obs_final[BETA][PREC] = trainer.mem_obs_final
            mem_total_reward[BETA][PREC] = trainer.mem_total_reward

    data = np.array((mem_obs_final, mem_total_reward))
    np.save(data_path, data)
else:
    data = np.load(data_path)
    mem_obs_final = data[0]
    mem_total_reward = data[1]

