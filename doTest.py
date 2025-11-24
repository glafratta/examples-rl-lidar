import agent
import gymnasium as gym
import numpy as np
import train
import plot
from gymnasium.envs.registration import register
import test

learning_rate=0.01
n_episodes=10000
start_epsilon=1.0
final_epsilon=0.1
epsilon_decay = start_epsilon / (n_episodes / 2) 

#register custom environment
register(
    id="LidarReading-c",
    entry_point="environment:LidarReading",
)
env=gym.make("LidarReading-c")
testing_agent=agent.Agent('qtable.npy')


test.test(agent, env)

