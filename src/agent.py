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
    - step_count tracks steps within current episode.
    """
    def __init__(self):
        self.step_count = 0  # Steps taken in current episode

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
        Resets step count.
        """
        self.step_count = 0

    def step(self):
        """
        Increment step count after each act(). Agents can use step_count for episode progress.
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
    Always returns a fixed action, or lowest/default for the action space.

    - If fixed_action is set, always returns it.
    - For Discrete: returns fixed_action_index (default 0).
    - For Box: returns action_space.low.
    """
    def __init__(self, action_space, fixed_action=None):
        super().__init__()
        self.action_space = action_space
        self.fixed_action = fixed_action
        self.fixed_action_index = 0  # Default for Discrete
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"

    def act(self, observation: Any) -> Any:
        if self.fixed_action is not None:
            action = self.fixed_action
            self.step()
            return action
        # Discrete action space: has 'n' attribute
        if hasattr(self.action_space, 'n'):
            n = getattr(self.action_space, 'n', None)
            idx = self.fixed_action_index
            if n is None:
                raise TypeError("action_space does not have 'n' attribute (not Discrete)")
            if not isinstance(idx, int):
                raise TypeError(f"fixed_action_index {idx} is not an integer")
            if 0 <= idx < n:
                action = idx
                self.step()
                return action
            else:
                raise ValueError(f"fixed_action_index {idx} out of bounds for Discrete(n={n})")
        # Box action space: has 'low' attribute
        elif hasattr(self.action_space, 'low'):
            low = getattr(self.action_space, 'low', None)
            if low is None:
                raise TypeError("action_space.low not found (not Box)")
            action = low
            self.step()
            return action
        else:
            raise NotImplementedError("DeterministicAgent only supports Discrete/Box action spaces or fixed_action")

    def set_fixed_action_index(self, index):
        self.fixed_action_index = index

class GreedyGridAgent(Agent):
    """
    Heuristic agent for SimpleGridWorldEnv: moves toward goal with tie-break preference (east, then north).
    Observation must be a string encoding position and goal.
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        # Parse position and goal from observation string
        pos = self._parse_position(observation)
        goal = self._parse_goal(observation)
        if pos is None or goal is None:
            action = self.action_space.sample()
            self.step()
            return action
        x, y = pos
        gx, gy = goal
        dx = gx - x
        dy = gy - y
        # Prefer east if dx > 0, west if dx < 0; north if dy < 0, south if dy > 0
        if dx != 0:
            if dx > 0:
                action = 2  # east
            else:
                action = 3  # west
        elif dy != 0:
            if dy < 0:
                action = 0  # north
            else:
                action = 1  # south
        else:
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
