"""
Base agent interface for RL environments. Concrete LLM-backed and heuristic agents subclass this.

Agents provide a unified interface for policy logic: observe the environment state and emit an action.
Subclasses implement 'act', and optionally manage internal state (via 'reset').

Current provided agents:
- RandomAgent: emits random actions from the action space.
- DeterministicAgent: always emits a fixed action or the lowest-valued action.
"""
from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """
    Abstract base class for RL agents.
    
    An agent observes the environment state and emits an action.
    Subclasses must implement 'act'.
    """
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
        """
        pass


class RandomAgent(Agent):
    """
    Agent that samples actions randomly from the environment's action space.
    Useful as a baseline for comparison.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        """
        Return a random action from the action space.
        """
        return self.action_space.sample()


class DeterministicAgent(Agent):
    """
    Agent that always returns a fixed action, or the lowest-valued action for the space.
    For discrete spaces, can specify index; for continuous, returns lowest value.
    """
    def __init__(self, action_space, fixed_action=None):
        """
        Args:
            action_space: The environment's action space.
            fixed_action: (optional) The action to always return. Must be contained in action_space.
        """
        self.action_space = action_space
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"
        self.fixed_action = fixed_action

    def act(self, observation: Any) -> Any:
        """
        Return the fixed action if specified, otherwise default to lowest-valued action.
        For discrete spaces: index 0 (or user-set index).
        For continuous spaces: lowest value.
        """
        if self.fixed_action is not None:
            return self.fixed_action
        # Default: always pick the lowest-valued action
        if hasattr(self.action_space, 'n'):
            # Allow user to specify index, else default to 0
            return getattr(self, 'fixed_action_index', 0)
        elif hasattr(self.action_space, 'low'):
            # For continuous spaces
            return self.action_space.low
        else:
            raise NotImplementedError("DeterministicAgent: unsupported action space type")

    def set_fixed_action_index(self, idx: int):
        """
        Set the action index for discrete action spaces.
        Args:
            idx: Integer index (must be in bounds for action_space.n)
        """
        if hasattr(self.action_space, 'n') and 0 <= idx < self.action_space.n:
            self.fixed_action_index = idx
        else:
            raise ValueError(f"Index {idx} out of bounds for discrete action space of size {getattr(self.action_space, 'n', None)}")
