"""Base agent interface. Concrete LLM-backed and heuristic agents subclass this."""
from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """An agent observes the env state and emits an action."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if internal state needs clearing."""
        pass


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        return self.action_space.sample()


class DeterministicAgent(Agent):
    def __init__(self, action_space, fixed_action=None):
        self.action_space = action_space
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"
        self.fixed_action = fixed_action

    def act(self, observation: Any) -> Any:
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
        if hasattr(self.action_space, 'n') and 0 <= idx < self.action_space.n:
            self.fixed_action_index = idx
        else:
            raise ValueError(f"Index {idx} out of bounds for discrete action space of size {getattr(self.action_space, 'n', None)}")
