import gymnasium as gym
import numpy as np

def train(n_episodes:int, agent, _env ):
      for episode in range(n_episodes):
      # Start a new hand
            obs, info = _env.reset()
            done = False

      # Play one complete hand
            while not done:
                  # Agent chooses action (initially random, gradually more intelligent)
                  action = agent.get_action(obs)

                  # Take action and observe result
                  next_obs, reward, terminated, truncated, info = _env.step(action)

                  # Learn from this experience
                  agent.update(obs, action, reward, terminated, next_obs)

                  # Move to next state
                  done = terminated or truncated
                  obs = next_obs

      # Reduce exploration rate (agent becomes less random over time)
      agent.decay_epsilon()