import agent
import environment
import gymnasium as gym

learning_rate=0.01
n_episodes=100000 
start_epsilon=1.0
final_epsilon=0.1
epsilon_decay = start_epsilon / (n_episodes / 2) 
env=gym.make("RaceTrack-c", render_mode="human")
agent=Agent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon)

def train(n_episodes:int, agent:Agent ):
      for episode in range(n_episodes):
      # Start a new hand
            obs, info = env.reset()
            done = False

      # Play one complete hand
            while not done:
                  # Agent chooses action (initially random, gradually more intelligent)
                  action = agent.get_action(obs)

                  # Take action and observe result
                  next_obs, reward, terminated, truncated, info = env.step(action)

                  # Learn from this experience
                  agent.update(obs, action, reward, terminated, next_obs)

                  # Move to next state
                  done = terminated or truncated
                  obs = next_obs

      # Reduce exploration rate (agent becomes less random over time)
      agent.decay_epsilon()

def main() -> int:
      return 0