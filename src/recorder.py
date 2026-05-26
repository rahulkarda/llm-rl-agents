import json
from typing import Any, Dict, Optional, List

class EpisodeRecorder:
    """
    Records episode transitions (state, action, reward, info) to a list and writes to JSONL.
    Also supports loading transitions from a JSONL file for replay.
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

    def load(self, path: Optional[str] = None):
        """
        Load transitions from a JSONL file into self.transitions.
        """
        in_file = path or self.out_path
        if in_file is None:
            raise ValueError("No input path specified for episode recorder.")
        self.transitions = []
        with open(in_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.transitions.append(json.loads(line))
