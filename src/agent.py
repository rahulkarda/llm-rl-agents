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
    - step_history: list of (observation, action) tuples for episode replay/debug.
    """
    def __init__(self):
        self.step_count = 0  # Number of actions taken in current episode
        self.step_history = []  # Stores (observation, action) per step in current episode

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
        self.step_history = []

    def step(self, observation=None, action=None):
        """
        Call after each act() to increment step_count within current episode.
        Optionally records (observation, action) to step_history for replay/debug.
        If observation and action are provided, appends to step_history.
        """
        self.step_count += 1
        if observation is not None and action is not None:
            self.step_history.append((observation, action))

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
    - New: Supports step-wise fixed action sequence (fixed_action_sequence).
    """
    def __init__(self, action_space, fixed_action=None):
        super().__init__()
        self.action_space = action_space
        self.fixed_action = fixed_action
        self.fixed_action_index = 0  # Default for Discrete
        self.fixed_box_action = None  # For Box action space
        self.fixed_action_sequence = None  # New: Sequence of actions
        self._sequence_cursor = 0     # Internal pointer for sequence
        if fixed_action is not None:
            assert self.action_space.contains(fixed_action), "fixed_action not in action_space"

    def act(self, observation: Any) -> Any:
        # Highest priority: step-wise action sequence (if set)
        if self.fixed_action_sequence is not None:
            if self._sequence_cursor >= len(self.fixed_action_sequence):
                # If out of actions, repeat last
                action = self.fixed_action_sequence[-1]
            else:
                action = self.fixed_action_sequence[self._sequence_cursor]
            self._sequence_cursor += 1
            return action
        # Next: fixed_action
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
            raise ValueError("fixed_action not in action_space")
        self.fixed_box_action = action

    def set_fixed_action_sequence(self, actions):
        """
        Set a sequence of actions to take, one per step. Resets sequence cursor.
        """
        self.fixed_action_sequence = actions
        self._sequence_cursor = 0

    def reset(self):
        """
        Reset agent state for new episode, including step count and action sequence cursor.
        """
        super().reset()
        self._sequence_cursor = 0

class GreedyGridAgent(Agent):
    """
    Moves toward goal in grid env using simple heuristics.

    Observation: str with format "Position: (x, y); Goal: (gx, gy); Obstacles: [(x1, y1), ...]"
    Action space: Discrete(4): 0=north, 1=south, 2=east, 3=west
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        # Parse observation string
        # Example: "Position: (2, 3); Goal: (7, 3); Obstacles: [(2, 5), (6, 2)]"
        pos = self._parse_position(observation)
        goal = self._parse_goal(observation)
        obstacles = self._parse_obstacles(observation)
        x, y = pos
        gx, gy = goal
        # Compute delta
        dx = gx - x
        dy = gy - y
        preferred = []
        # Prefer east, then south, then north, then west (if not blocked)
        if dx > 0:
            preferred.append(2)  # east
        elif dx < 0:
            preferred.append(3)  # west
        if dy > 0:
            preferred.append(1)  # south
        elif dy < 0:
            preferred.append(0)  # north
        # Tie-break: east > south > north > west
        for a in [2, 1, 0, 3]:
            if a not in preferred:
                preferred.append(a)
        # Avoid obstacles if possible
        for action in preferred:
            nx, ny = self._next_position(x, y, action)
            if (nx, ny) not in obstacles:
                return action
        # If all moves blocked, return random valid action
        return self.action_space.sample()

    def _parse_position(self, obs_str):
        m = re.search(r"Position: \((\d+),\s*(\d+)\)", obs_str)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    def _parse_goal(self, obs_str):
        m = re.search(r"Goal: \((\d+),\s*(\d+)\)", obs_str)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    def _parse_obstacles(self, obs_str):
        m = re.search(r"Obstacles: \[(.*)\]", obs_str)
        obstacles_str = m.group(1) if m else ''
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
