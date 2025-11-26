import numpy as np
import gymnasium as gym

def test(agent, env, n_episodes=1000):
      total_rewards=[]
       # Temporarily disable exploration for testing
      old_epsilon = agent.epsilon
      agent.epsilon = 0.0  # Pure exploitation

      for _ in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            s=0
            while not done:
                  obs_tuple=((obs['agent'][0],obs['agent'][0]), (obs['target'][0], obs['target'][1]))
                  action = agent.get_action(obs_tuple)
                  obs, reward, terminated, truncated, info = env.step(action)
                 # env.render()
                  episode_reward += reward
                  done = terminated or truncated
                  s=s+1
            #print("%i terminated =%i, truncated %i", s, terminated, truncated )
            total_rewards.append(episode_reward)

      # Restore original epsilon
      agent.epsilon = old_epsilon

      success_rate = np.mean(np.array(total_rewards) > 0)
      average_reward = np.mean(total_rewards)
      return success_rate, average_reward
