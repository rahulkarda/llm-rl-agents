"""
Agent interfaces for RL environments. Provides abstract base and concrete agent implementations.

Agents observe the environment and emit an action. Subclass Agent and implement act() for your policy logic.

Provided agents:
- RandomAgent: samples actions randomly from action space.
- DeterministicAgent: always returns a fixed or default action.
- GreedyGridAgent: moves toward goal in grid env using heuristics.

Usage:
    agent = RandomAgent(env.action_space)
    action = agent.act(observation)

Extension:
    - Subclass Agent and implement act(observation).
    - Optionally override reset() for internal state.
"""
from abc import ABC, abstractmethod
from typing import Any
import re

class Agent(ABC):
    """
    Abstract base class for RL agents.

    - Observe environment state, emit action via act(observation).
    - Optionally implement reset() to clear state between episodes.
    - step_count tracks number of actions taken in current episode.
    """
    def __init__(self):
        self.step_count = 0  # Number of actions taken in current episode

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Compute and return an action given the current observation.
        Args:
            observation: Environment state (arbitrary type).
        Returns:
            action: Action to take (type depends on environment).
        """
        ...

    def reset(self) -> None:
        """
        Called at episode start. Override to clear internal state if needed.
        Resets step_count to zero for new episode.
        """
        self.step_count = 0

    def step(self):
        """
        Call after each act() to increment step_count within current episode.
        """
        self.step_count += 1

class RandomAgent(Agent):
    """
    Samples random actions from environment's action space. Useful as a baseline.
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        action = self.action_space.sample()
        return action

class DeterministicAgent(Agent):
    """
    Always returns a fixed action, or default for the action space.

    - If fixed_action is set, always returns it.
    - For Discrete: returns fixed_action_index (default 0).
    - For Box: returns action_space.low.
    """
    def __init__(self, action_space, fixed_action=None):
        super().__init__()
        self.action_space = action_space
        self.fixed_action = fixed_action
        self.fixed_action_index = 0  # Default for Discrete
        self.fixed_box_action = None  # New: For Box action space
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"

    def act(self, observation: Any) -> Any:
        # Highest priority: fixed_action
        if self.fixed_action is not None:
            action = self.fixed_action
            return action
        # Discrete action space
        if hasattr(self.action_space, 'n'):
            n = self.action_space.n
            idx = self.fixed_action_index
            if not isinstance(idx, int):
                raise TypeError(f"fixed_action_index {idx} is not an integer")
            if 0 <= idx < n:
                action = idx
                return action
            else:
                raise ValueError(f"fixed_action_index {idx} out of bounds for Discrete(n={n})")
        # Box action space
        elif hasattr(self.action_space, 'low'):
            if self.fixed_box_action is not None:
                action = self.fixed_box_action
            else:
                action = self.action_space.low
            return action
        else:
            raise NotImplementedError("DeterministicAgent only supports Discrete/Box action spaces or fixed_action")

    def set_fixed_action_index(self, index):
        """
        Set the action index for Discrete action spaces. Does not affect fixed_action.
        """
        self.fixed_action_index = index

    def set_fixed_action(self, action):
        """
        Set the fixed action for Box action spaces (or any custom action).
        Only used if fixed_action is None. Checks that action is valid for action_space.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"fixed_action {action} not in action_space")
        self.fixed_box_action = action

class GreedyGridAgent(Agent):
    """
    Heuristic agent for SimpleGridWorldEnv: moves toward goal with tie-break preference (east, then south, then north).
    Optionally supports diagonal move preference if diagonal_preference=True.
    Observation must be a string encoding position and goal.
    Now supports obstacle avoidance if obstacles are listed in observation.
    """
    def __init__(self, action_space, diagonal_preference=False):
        super().__init__()
        self.action_space = action_space
        self.diagonal_preference = diagonal_preference

    def act(self, observation: Any) -> Any:
        # Observation: 'pos=(x,y);goal=(gx,gy);obstacles=[(x1,y1),...]'
        obs_info = self._parse_obs(observation)
        x, y = obs_info['x'], obs_info['y']
        gx, gy = obs_info['gx'], obs_info['gy']
        obstacles = obs_info['obstacles']
        dx = gx - x
        dy = gy - y
        # Diagonal preference: move both axes if possible
        if self.diagonal_preference:
            if dx != 0 and dy != 0:
                action = 2 if dx > 0 else 3  # east/west
                nx, ny = self._next_position(x, y, action)
                if (nx, ny) not in obstacles:
                    return action
                action = 0 if dy < 0 else 1  # north/south
                nx, ny = self._next_position(x, y, action)
                if (nx, ny) not in obstacles:
                    return action
        # Regular tie-break: east, south, north, west
        candidates = []
        if dx > 0:
            candidates.append(2)  # east
        elif dx < 0:
            candidates.append(3)  # west
        if dy > 0:
            candidates.append(1)  # south
        elif dy < 0:
            candidates.append(0)  # north
        # Tie-break: east, then south, then north, then west
        preferred_order = [2, 1, 0, 3]
        for action in preferred_order:
            if action in candidates:
                nx, ny = self._next_position(x, y, action)
                if (nx, ny) not in obstacles:
                    return action
        # If all blocked or at goal, pick random valid
        for action in preferred_order:
            nx, ny = self._next_position(x, y, action)
            if (nx, ny) not in obstacles:
                return action
        return 0  # fallback: north

    def _parse_obs(self, obs_str):
        # Example: 'pos=(3,2);goal=(7,7);obstacles=[(2,3),(4,5)]'
        pos_match = re.search(r"pos=\((\d+),(\d+)\)", obs_str)
        goal_match = re.search(r"goal=\((\d+),(\d+)\)", obs_str)
        obstacles_match = re.search(r"obstacles=\[(.*?)\]", obs_str)
        x = int(pos_match.group(1)) if pos_match else 0
        y = int(pos_match.group(2)) if pos_match else 0
        gx = int(goal_match.group(1)) if goal_match else 0
        gy = int(goal_match.group(2)) if goal_match else 0
        obstacles = self._parse_obstacles(obstacles_match.group(1) if obstacles_match else "None")
        return {'x': x, 'y': y, 'gx': gx, 'gy': gy, 'obstacles': obstacles}

    def _parse_obstacles(self, obstacles_str):
        if obstacles_str.strip() == '' or obstacles_str.strip() == 'None':
            return []
        tuples = re.findall(r"\((\d+),\s*(\d+)\)", obstacles_str)
        return [(int(x), int(y)) for x, y in tuples]

    def _next_position(self, x, y, action):
        # Action: 0=north, 1=south, 2=east, 3=west
        if action == 0:
            return x, y - 1
        elif action == 1:
            return x, y + 1
        elif action == 2:
            return x + 1, y
        elif action == 3:
            return x - 1, y
        else:
            return x, y
