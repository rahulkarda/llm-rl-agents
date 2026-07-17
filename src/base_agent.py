from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):
    """
    Abstract base class for RL agents.
    
    Agents observe the environment state and emit an action.
    Subclasses must implement 'act'.
    Use as a base for all agent types (random, heuristic, LLM, etc).

    Usage:
        - Subclass Agent and implement act(observation) -> action.
        - Optionally override reset() if agent maintains internal state.
        - Call reset() at start of each episode.
    
    Extension notes:
        - The observation type is arbitrary and depends on the environment (e.g., array, dict, string).
        - The action type must be compatible with the environment's action_space.
        - Agents that keep internal state (e.g., counters, history) should override reset() to clear state between episodes.
        - For stateless agents, reset() can be a no-op.
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
        Default: no-op.
        """
        pass
