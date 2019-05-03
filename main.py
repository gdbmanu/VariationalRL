from TP1_environment import Environment
from agent import Agent, Q_learning_trainer, one_step_variational_trainer, final_variational_trainer

# Hyperparameters

N = 8000 # Number of episodes

# Agent/Environment initialization

env = Environment()
agent = Agent(GAMMA = 1)
#trainer = Q_learning_trainer(agent)
#trainer = one_step_variational_trainer(agent)
trainer = final_variational_trainer(agent)

for i in range(N):
    trainer.run_episode()
    print("Trajectory: ", trainer.trajectory)
    print("Total reward got: %.4f" % trainer.total_reward)

print('Q_ref', trainer.Q_ref)
print('Q_KL_diff', trainer.Q_KL_diff)
print('Q', agent.Q)
print(trainer.nb_visits)
print(sum(trainer.obs_score))
#while not env.is_done():
#    agent.step(env)

