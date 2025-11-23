import numpy as np
import torch
import gymnasium as gym
from abc import ABC, abstractmethod

#making a custom env

class LidarReading(gym.Env): #continuous state-space
    def __init__(self, size: int=110): #110x110 grid where each square is 1cm
        self.size=size
        self._agent_location = np.array([-1, -1], dtype=np.float32) #this initialises agent location and target but need to override
        self._target_location = np.array([-1, -1], dtype=np.float32) #the random initialisation in reset

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=float),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=float),  # [x, y] coordinates
            }
        )

        self.action_space = gym.spaces.Discrete(3) #left, right, default

        self._action_to_direction = {
            0: np.array([0.01, 0]),   # Move right (positive x)
            1: np.array([0, 0.01]),   # Move up (positive y)
            2: np.array([-0.01, 0]),  # Move left (negative x)
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def reset(self, *, seed = None, options = None):
        return super().reset(seed=seed, options=options)
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=float)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=float
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        reward = 1 if terminated else 0