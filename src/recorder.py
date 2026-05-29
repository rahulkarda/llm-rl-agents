import json
from typing import Any, Dict, Optional, List

class EpisodeRecorder:
    """
    Store episode transitions (observation, action, reward, info) in memory.
    Supports saving transitions to JSONL, loading for replay, and computing episode summary statistics.
    Optionally caps the number of transitions buffered for memory efficiency.
    """
    def __init__(self, out_path: Optional[str] = None, max_transitions: Optional[int] = None):
        self.out_path = out_path
        self.transitions: List[Dict[str, Any]] = []
        self.max_transitions = max_transitions

    def record_transition(self, observation: Any, action: Any, reward: float, info: Optional[Dict[str, Any]] = None):
        """
        Add a single transition to the buffer.
        Args:
            observation: Environment observation/state.
            action: Action taken by agent.
            reward: Reward obtained.
            info: Optional info dict from environment.
        """
        if self.max_transitions is not None and len(self.transitions) >= self.max_transitions:
            # Drop oldest transition to keep within cap
            self.transitions.pop(0)
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            transition["info"] = info
        self.transitions.append(transition)

    def clear(self):
        """
        Empty the transition buffer.
        """
        self.transitions = []

    def save_to_jsonl(self, path: Optional[str] = None):
        """
        Save buffered transitions to a JSONL file.
        Args:
            path: Optional override path to save file.
        """
        target_path = path or self.out_path
        if target_path is None:
            raise ValueError("No output path specified for episode recorder.")
        with open(target_path, 'w', encoding='utf-8') as f:
            for transition in self.transitions:
                json.dump(transition, f)
                f.write('\n')

    def load_from_jsonl(self, path: Optional[str] = None):
        """
        Load transitions from a JSONL file into the buffer.
        Args:
            path: Optional override path to load file.
        """
        source_path = path or self.out_path
        if source_path is None:
            raise ValueError("No input path specified for episode recorder.")
        self.transitions = []
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.transitions.append(json.loads(line))
        if self.max_transitions is not None and len(self.transitions) > self.max_transitions:
            self.transitions = self.transitions[-self.max_transitions:]

    def episode_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics from the episode transitions.
        Returns:
            dict with keys: 'length', 'total_reward', 'average_reward', 'actions'
        """
        length = len(self.transitions)
        total_reward = sum(t.get('reward', 0.0) for t in self.transitions)
        average_reward = total_reward / length if length > 0 else 0.0
        actions = [t.get('action') for t in self.transitions]
        return {
            'length': length,
            'total_reward': total_reward,
            'average_reward': average_reward,
            'actions': actions
        }
