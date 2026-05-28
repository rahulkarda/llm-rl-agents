import json
from typing import Any, Dict, Optional, List

class EpisodeRecorder:
    """
    Records episode transitions (observation, action, reward, info) to an internal list.
    Supports saving transitions to JSONL and loading for replay. Provides summary statistics for episodes.
    """
    def __init__(self, out_path: Optional[str] = None):
        self.out_path = out_path
        self.transitions: List[Dict[str, Any]] = []

    def record(self, observation: Any, action: Any, reward: float, info: Optional[Dict[str, Any]] = None):
        """
        Append a single transition to the internal buffer.
        """
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            transition["info"] = info
        self.transitions.append(transition)

    def reset(self):
        """
        Clear all buffered transitions.
        """
        self.transitions = []

    def save(self, path: Optional[str] = None):
        """
        Write buffered transitions to a JSONL file.
        """
        target_path = path or self.out_path
        if target_path is None:
            raise ValueError("No output path specified for episode recorder.")
        with open(target_path, 'w', encoding='utf-8') as f:
            for transition in self.transitions:
                json.dump(transition, f)
                f.write('\n')

    def load(self, path: Optional[str] = None):
        """
        Load transitions from a JSONL file into the buffer.
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

    def summary(self) -> Dict[str, Any]:
        """
        Compute episode summary statistics from transitions.
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
