from environment import Environment
from agent import Agent, Q_learning_trainer, one_step_variational_trainer, final_variational_trainer

# Hyperparameters

N = 10000 # Number of episodes

# Agent/Environment initialization

#env = Environment.tp1(initial_state_range=4)
env = Environment.bottleneck(initial_state_range=0)
agent = Agent.timeAgent(env, GAMMA=1, BETA=10, do_reward = True)

#trainer = Q_learning_trainer(agent)
#trainer = one_step_variational_trainer(agent)
trainer = final_variational_trainer(agent, EPSILON = 1e-3, ref_prob = 'unif')
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

