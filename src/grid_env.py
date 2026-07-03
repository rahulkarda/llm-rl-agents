"""
Simple text-based grid world environment for RL agent testing.

Design:
- Agent starts at (0,0) on a N x N grid.
- Actions: north, south, east, west.
- Reward: +1 for reaching goal (bottom-right), else 0.
- Observation: string describing position and grid.
- Max steps: configurable (default 100).
- Step returns obs (str), reward (float), done (bool), truncated (bool), info (dict).

Usage example:
    from grid_env import SimpleGridWorldEnv
    env = SimpleGridWorldEnv(grid_size=8, max_steps=100)
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs2, reward, done, truncated, info = env.step(action)
        print(obs2, reward, done)

Design notes:
- Observation is always a string for easy LLM prompting.
- Action space: Discrete(4) (0=north, 1=south, 2=east, 3=west).
- Goal: reach cell (N-1, N-1).
- Truncated: True if max_steps reached.
- info dict includes: step, position, goal, grid_size.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleGridWorldEnv(gym.Env):
    """
    Simple N x N grid world. Agent moves in four directions to reach goal.
    Observation: text describing position and grid.
    Action space: Discrete(4) (north, south, east, west).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=8, max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # 0=north, 1=south, 2=east, 3=west
        self.observation_space = spaces.Text(max_length=128)
        self.goal = (grid_size - 1, grid_size - 1)
        self.position = (0, 0)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.position = (0, 0)
        self.step_count = 0
        obs = self._get_obs()
        info = {"step": self.step_count, "position": self.position, "goal": self.goal, "grid_size": self.grid_size}
        return obs, info

    def step(self, action):
        """
        action: 0=north, 1=south, 2=east, 3=west
        """
        x, y = self.position
        if action == 0 and x > 0:
            x -= 1  # north
        elif action == 1 and x < self.grid_size - 1:
            x += 1  # south
        elif action == 2 and y < self.grid_size - 1:
            y += 1  # east
        elif action == 3 and y > 0:
            y -= 1  # west
        # else: invalid move, stay
        self.position = (x, y)
        self.step_count += 1
        reward = 1.0 if self.position == self.goal else 0.0
        done = self.position == self.goal
        truncated = self.step_count >= self.max_steps and not done
        obs = self._get_obs()
        info = {
            "step": self.step_count,
            "position": self.position,
            "goal": self.goal,
            "grid_size": self.grid_size
        }
        return obs, reward, done, truncated, info

    def _get_obs(self):
        """
        Returns text describing current position and grid.
        """
        x, y = self.position
        obs = f"You are at position ({x}, {y}) in a {self.grid_size}x{self.grid_size} grid. "
        obs += f"Goal is at ({self.goal[0]}, {self.goal[1]})."
        return obs

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        x, y = self.position
        gx, gy = self.goal
        grid[gx, gy] = "G"
        grid[x, y] = "A"
        lines = [" ".join(row) for row in grid]
        print("\n".join(lines))

    def close(self):
        pass
