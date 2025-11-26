import agent
import gymnasium as gym
import numpy as np
import train
import test
import plot
from gymnasium.envs.registration import register


learning_rate=0.01
n_episodes=5000
start_epsilon=1.0
final_epsilon=0.1
epsilon_decay = start_epsilon / (n_episodes / 2) 

#register custom environment
register(
    id="LidarReading-c",
    entry_point="environment:LidarReading",
)
env=gym.make("LidarReading-c", _render="human")
env.reset()
training_agent=agent.Agent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon)

train.train(n_episodes, training_agent, env)
#np.save("q_values.npy", training_agent.q_values) #dump q valuess
result=test.test(training_agent, env)
print(result)
#plot.plot(env,training_agent)

