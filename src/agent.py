"""
Base agent interface for RL environments. Concrete LLM-backed and heuristic agents subclass this.

Agents provide a unified interface for policy logic: observe the environment state and emit an action.
Subclasses implement 'act', and optionally manage internal state (via 'reset').

Current provided agents:
- RandomAgent: emits random actions from the action space.
- DeterministicAgent: always emits a fixed action or the lowest-valued action.
- GreedyGridAgent: moves toward goal in SimpleGridWorldEnv (heuristic).

Usage:
    # Instantiate an agent with env.action_space
    agent = RandomAgent(env.action_space)
    observation = ...
    action = agent.act(observation)

Extension notes:
    - To make a new agent, subclass Agent and implement act().
    - For LLM-based policies, override act() to prompt the model and parse the action.
    - reset() can be overridden to clear internal state between episodes.
"""
from abc import ABC, abstractmethod
from typing import Any
import re


class Agent(ABC):
    """
    Abstract base class for RL agents.
    
    An agent observes the environment state and emits an action.
    Subclasses must implement 'act'.
    Provides step count tracking for episode progress.
    """
    def __init__(self):
        self.step_count = 0  # Tracks number of steps in current episode

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Compute and return an action given the current observation.
        Args:
            observation: Environment state (arbitrary type).
        Returns:
            action: Action to take (type depends on env).
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode. Override if internal state needs clearing.
        Resets step count.
        """
        self.step_count = 0

    def step(self):
        """
        Call this method after each act() to increment step count.
        Agents can use step_count for episode progress awareness.
        """
        self.step_count += 1


class RandomAgent(Agent):
    """
    Agent that samples actions randomly from the environment's action space.
    Useful as a baseline for comparison.
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        """
        Return a random action from the action space.
        """
        action = self.action_space.sample()
        self.step()
        return action


class DeterministicAgent(Agent):
    """
    Agent that always returns a fixed action, or selects a default for the action space.

    Action selection:
      - If fixed_action is specified: always return that.
      - For Discrete action spaces: return fixed_action_index (default 0).
      - For Box action spaces: return lowest value (action_space.low).
      - Otherwise: NotImplementedError.

    For Discrete spaces, fixed_action_index (default 0) can be set via set_fixed_action_index().
    For Box spaces, returns action_space.low.
    """
    def __init__(self, action_space, fixed_action=None):
        """
        Args:
            action_space: The environment's action space.
            fixed_action: (optional) The action to always return. Must be contained in action_space.
        """
        super().__init__()
        self.action_space = action_space
        self.fixed_action = fixed_action
        self.fixed_action_index = 0  # Default for Discrete
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"

    def act(self, observation: Any) -> Any:
        """
        Return deterministic action based on agent setup and action space:
            - If fixed_action is set: always return it.
            - Else, for Discrete: return fixed_action_index (default 0).
            - Else, for Box: return action_space.low.
            - Else: raise NotImplementedError.
        """
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
                raise TypeError("action_space does not have 'low' attribute (not Box)")
            action = low
            self.step()
            return action
        else:
            raise NotImplementedError("DeterministicAgent only supports Discrete and Box action spaces.")

    def set_fixed_action_index(self, idx):
        self.fixed_action_index = idx


class GreedyGridAgent(Agent):
    """
    Heuristic agent for SimpleGridWorldEnv.
    Moves toward goal using Manhattan distance, preferring east before south.
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        # Observation: text with position and grid info
        pos = self._parse_position(observation)
        goal = self._parse_goal(observation)
        if pos is None or goal is None:
            return self.action_space.sample()  # fallback
        x, y = pos
        gx, gy = goal
        if x < gx:
            action = 1  # south
        elif y < gy:
            action = 2  # east
        elif y > gy:
            action = 3  # west
        elif x > gx:
            action = 0  # north
        else:
            action = self.action_space.sample()  # already at goal, random action
        self.step()
        return action

    def _parse_position(self, obs_str):
        # Find 'position (x, y)'
        m = re.search(r"position \((\d+), (\d+)\)", obs_str)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _parse_goal(self, obs_str):
        # Find 'Goal is at (gx, gy)'
        m = re.search(r"Goal is at \((\d+), (\d+)\)", obs_str)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

