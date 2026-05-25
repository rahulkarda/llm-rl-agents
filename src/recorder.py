import json
from typing import Any, Dict, Optional, List

class EpisodeRecorder:
    """
    Records episode transitions (state, action, reward, info) to a list and writes to JSONL.
    """
    def __init__(self, out_path: Optional[str] = None):
        self.out_path = out_path
        self.transitions: List[Dict[str, Any]] = []

    def record(self, observation: Any, action: Any, reward: float, info: Optional[Dict[str, Any]] = None):
        entry = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            entry["info"] = info
        self.transitions.append(entry)

    def reset(self):
        self.transitions = []

    def save(self, path: Optional[str] = None):
        """
        Write all transitions to a JSONL file.
        """
        out_file = path or self.out_path
        if out_file is None:
            raise ValueError("No output path specified for episode recorder.")
        with open(out_file, 'w', encoding='utf-8') as f:
            for t in self.transitions:
                json.dump(t, f)
                f.write('\n')
