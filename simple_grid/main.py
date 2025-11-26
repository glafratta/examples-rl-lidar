import agent
import gymnasium as gym
import numpy as np
import train
import test
import plot
from gymnasium.envs.registration import register


learning_rate=0.01
n_episodes=100000
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
result=test.test(training_agent, env)
print(result)

