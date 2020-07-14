import gym
import roboschool
from agent import Agent
from trainer import Final_variational_trainer, Q_learning_trainer
import numpy as np

ENV_NAME = 'RoboschoolAnt-v1' #'MountainCar-v0' #'FrozenLake-v0' #
env = gym.make(ENV_NAME)

GAMMA=0.9
OBS_LEAK = 1e-6
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

data_path = '200714-Ant-final-with-reward-DQN-Adam.npy'
BETA_range = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
PREC_range = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

if not os.path.isfile(data_path):
    mem_obs_final = {}
    mem_total_reward = {}
    mem_pred_reward = {}
    mem_pred_var = {}

    tic = time.clock()

    for trial in range(10):

        mem_obs_final[trial] = {}
        mem_total_reward[trial] = {}
        mem_pred_reward[trial] = {}
        mem_pred_var[trial] = {}

        for BETA in BETA_range:

            mem_obs_final[trial][BETA] = {}
            mem_total_reward[trial][BETA] = {}
            mem_pred_reward[trial][BETA] = {}
            mem_pred_var[trial][BETA] = {}

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
                              offPolicy=offPolicy,
                              optim='Adam')
                trainer = Final_variational_trainer(agent,
                                                    monte_carlo=monte_carlo,
                                                    augmentation=augmentation,
                                                    final=final,
                                                    OBS_LEAK=OBS_LEAK,
                                                    ref_prob='unif',
                                                    HIST_HORIZON=200 * int(1 / OBS_LEAK),
                                                    N_PART=300)
                for i in range(N):
                    # print(i)
                    trainer.run_episode()
                    if (i + 1) % 1000 == 0:
                        # plt.figure(figsize = (4, 4))
                        # plt.plot(agent.KL.flatten())
                        print(trainer.nb_trials)
                        # print("Trajectory: ", trainer.trajectory)
                        print("Total reward got: %.4f" % trainer.total_reward)

                obs = trainer.mem_obs_final[-1]*0
                act = trainer.action_history[-1]*0

                mem_obs_final[trial][BETA][PREC] = trainer.mem_obs_final
                mem_total_reward[trial][BETA][PREC] = trainer.mem_total_reward

                pred_reward = 0 #trainer.calc_sum_future_rewards(0, obs, done=False)
                mem_pred_reward[trial][BETA][PREC] = pred_reward

                pred_var = agent.Q_var(obs, act) #trainer.agent.softmax_expectation(obs, trainer.agent.set_Q_obs(obs))
                mem_pred_var[trial][BETA][PREC] = pred_var

                data = np.array((mem_obs_final, mem_total_reward, mem_pred_reward, mem_pred_var))
                np.save(data_path, data)
