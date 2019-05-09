from environment import Environment
from agent import Agent
from trainer import Q_learning_trainer, One_step_variational_trainer, Final_variational_trainer

# Hyperparameters

N = 10000 # Number of episodes

# Agent/Environment initialization

#env = Environment.tp1(initial_state_range=4)
#env = Environment.bottleneck(initial_state_range=0)
env = Environment.square(initial_state_range=0)

agent = Agent(env, GAMMA=1, ALPHA= 3e-2, BETA=8, do_reward = False)
#agent = Agent.timeAgent(env, GAMMA=1, ALPHA= 3e-2, BETA=8, do_reward = True)

trainer = Q_learning_trainer(agent)
#trainer = One_step_variational_trainer(agent)
#trainer = Final_variational_trainer(agent, EPSILON = 1e-3, ref_prob = 'unif')
#
for i in range(N):
    print(i)
    trainer.run_episode()
    print("Trajectory: ", trainer.trajectory)
    print("Total reward got: %.4f" % trainer.total_reward)
#
print('Q_ref', agent.Q_ref)
print('KL', agent.KL)
print('Q_var', agent.Q_var)
print(trainer.nb_visits_final)
print(trainer.obs_score_final)
#while not env.is_done():
#    agent.step(env)

