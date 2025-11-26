import gymnasium as gym
import numpy as np

def train(n_episodes:int, agent, _env ):
      print("Training")
      for episode in range(n_episodes):
            print("Episode ", episode)
      # Start a new hand
            obs, info = _env.reset()
            done = False

      # Play one complete hand
            while not done:
                  # Agent chooses action (initially random, gradually more intelligent)
                  action = agent.get_action(obs)

                  # Take action and observe result
                  next_obs, reward, terminated, truncated, info = _env.step(action)
                  #_env.render()
                  
                  obs_tuple=((obs['agent'][0],obs['agent'][0]), (obs['target'][0], obs['target'][1]))
                  next_obs_tuple=((next_obs['agent'][0],next_obs['agent'][0]), (next_obs['target'][0], next_obs['target'][1]))

                  # Learn from this experience
                  agent.update(obs_tuple, action, reward, terminated, next_obs_tuple)

                  # Move to next state
                  done = terminated or truncated
                  obs = next_obs

      # Reduce exploration rate (agent becomes less random over time)
      agent.decay_epsilon()