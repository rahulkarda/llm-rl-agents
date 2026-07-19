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
        self.step()
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
            self.step()
            return action
        # Discrete action space
        if hasattr(self.action_space, 'n'):
            n = self.action_space.n
            idx = self.fixed_action_index
            if not isinstance(idx, int):
                raise TypeError(f"fixed_action_index {idx} is not an integer")
            if 0 <= idx < n:
                action = idx
                self.step()
                return action
            else:
                raise ValueError(f"fixed_action_index {idx} out of bounds for Discrete(n={n})")
        # Box action space
        elif hasattr(self.action_space, 'low'):
            if self.fixed_box_action is not None:
                action = self.fixed_box_action
            else:
                action = self.action_space.low
            self.step()
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
        obs_str = str(observation)
        pos = self._parse_position(obs_str)
        goal = self._parse_goal(obs_str)
        obstacles = self._parse_obstacles(obs_str)
        if pos is None or goal is None:
            action = self.action_space.sample()
        else:
            x, y = pos
            gx, gy = goal
            move_candidates = []
            # Diagonal preference
            if self.diagonal_preference and x != gx and y != gy:
                if gx > x and gy > y:
                    move_candidates.append(2)  # east
                    move_candidates.append(1)  # south
                elif gx < x and gy < y:
                    move_candidates.append(3)  # west
                    move_candidates.append(0)  # north
                elif gx > x and gy < y:
                    move_candidates.append(2)  # east
                    move_candidates.append(0)  # north
                elif gx < x and gy > y:
                    move_candidates.append(3)  # west
                    move_candidates.append(1)  # south
            # Normal axis-aligned moves
            else:
                if gx > x:
                    move_candidates.append(2)  # east
                elif gx < x:
                    move_candidates.append(3)  # west
                if gy > y:
                    move_candidates.append(1)  # south
                elif gy < y:
                    move_candidates.append(0)  # north
            # Tie-break order: east, south, north, west
            for a in [2, 1, 0, 3]:
                if a not in move_candidates:
                    move_candidates.append(a)
            # Try moves in order, avoid obstacles
            action = None
            for candidate in move_candidates:
                nx, ny = self._next_position(x, y, candidate)
                if obstacles is not None and (nx, ny) in obstacles:
                    continue
                action = candidate
                break
            if action is None:
                action = self.action_space.sample()
        self.step()
        return action

    def _parse_position(self, obs_str):
        m = re.search(r"position \((\d+), (\d+)\)", obs_str)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _parse_goal(self, obs_str):
        m = re.search(r"Goal is at \((\d+), (\d+)\)", obs_str)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _parse_obstacles(self, obs_str):
        # Looks for 'Obstacles: [(x1, y1), (x2, y2), ...]'
        m = re.search(r"Obstacles: \[(.*?)\]", obs_str)
        if m:
            obstacles_str = m.group(1)
            # Find all (x, y) tuples
            tuples = re.findall(r"\((\d+),\s*(\d+)\)", obstacles_str)
            return [(int(x), int(y)) for x, y in tuples]
        return None

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
