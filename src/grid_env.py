import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleGridWorldEnv(gym.Env):
    """
    Minimal TextWorld-style grid game wrapper.
    Grid is NxN, agent starts in random cell, goal is in random cell.
    Observations: textual description of current position and surroundings.
    Actions: ["north", "south", "east", "west"]
    Rewards: +1 for reaching goal, 0 otherwise.
    Episode ends when goal is reached or max_steps is exceeded.
    """
    def __init__(self, grid_size=7, max_steps=45):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # 0: north, 1: south, 2: east, 3: west
        self.observation_space = spaces.Text()  # textual observation
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0
        self._rng = np.random.default_rng()

    ACTIONS = ["north", "south", "east", "west"]

    def reset(self, seed=None, options=None):
        # Properly seed the environment
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        self.agent_pos = [self._rng.integers(self.grid_size), self._rng.integers(self.grid_size)]
        while True:
            self.goal_pos = [self._rng.integers(self.grid_size), self._rng.integers(self.grid_size)]
            if self.goal_pos != self.agent_pos:
                break
        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Map action index to direction
        if isinstance(action, str):
            try:
                action_idx = self.ACTIONS.index(action.lower())
            except Exception:
                action_idx = self._rng.integers(4)
        else:
            action_idx = int(action)

        old_pos = self.agent_pos.copy()
        if action_idx == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1  # north
        elif action_idx == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1  # south
        elif action_idx == 2 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1  # east
        elif action_idx == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1  # west
        # else: invalid move, stay

        self.steps += 1
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else 0.0
        obs = self._get_obs()
        info = {
            "agent_pos": tuple(self.agent_pos),
            "goal_pos": tuple(self.goal_pos),
            "step": self.steps,
            "last_action": self.ACTIONS[action_idx],
            "moved": old_pos != self.agent_pos,
        }
        return obs, reward, done, False, info

    def _get_obs(self):
        desc = f"You are at position {self.agent_pos[0]+1},{self.agent_pos[1]+1} in a {self.grid_size}x{self.grid_size} grid. "
        if self.agent_pos == self.goal_pos:
            desc += "You see the goal here!"
        else:
            desc += f"The goal is somewhere else. "
        return desc
    
    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[self.goal_pos[0], self.goal_pos[1]] = "G"
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        out = "\n".join([" ".join(row) for row in grid])
        print(out)
