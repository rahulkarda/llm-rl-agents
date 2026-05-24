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
