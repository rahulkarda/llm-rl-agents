from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):
    """
    Abstract base class for RL agents.
    Agents observe environment state and emit an action.
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
        Called at episode start. Override if agent maintains internal state.
        """
        pass
