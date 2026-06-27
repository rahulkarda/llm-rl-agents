"""
Base agent interface for RL environments. Concrete LLM-backed and heuristic agents subclass this.

Agents provide a unified interface for policy logic: observe the environment state and emit an action.
Subclasses implement 'act', and optionally manage internal state (via 'reset').

Current provided agents:
- RandomAgent: emits random actions from the action space.
- DeterministicAgent: always emits a fixed action or the lowest-valued action.

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
        return self.action_space.sample()


class DeterministicAgent(Agent):
    """
    Agent that always returns a fixed action, or defaults to the lowest-valued action for the space.
    
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
            return self.fixed_action

        # Discrete action space: has 'n' attribute
        if hasattr(self.action_space, 'n'):
            # Ensure index is in bounds
            if 0 <= self.fixed_action_index < self.action_space.n:
                return self.fixed_action_index
            else:
                raise ValueError(f"fixed_action_index {self.fixed_action_index} out of bounds for Discrete(n={self.action_space.n})")
        # Box action space: has 'low' attribute
        elif hasattr(self.action_space, 'low'):
            return self.action_space.low
        else:
            raise NotImplementedError("DeterministicAgent: unsupported action space type")

    def set_fixed_action_index(self, idx: int):
        """
        Set the action index for discrete action spaces.
        Args:
            idx: Integer index (must be in bounds for action_space.n)
        """
        if hasattr(self.action_space, 'n'):
            if 0 <= idx < self.action_space.n:
                self.fixed_action_index = idx
            else:
                raise ValueError(f"Index {idx} out of bounds for discrete action space of size {self.action_space.n}")
        else:
            raise TypeError("set_fixed_action_index: action_space is not Discrete")
