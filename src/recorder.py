import json
from typing import Any, Dict, Optional, List, Callable
import csv
from utils import flatten_dict

class EpisodeRecorder:
    """
    Stores episode transitions (observation, action, reward, info) for RL episodes.
    Supports saving transitions to JSONL for debugging/replay, loading episode traces, and computing summary statistics.
    Self-reflection loop scaffold: critique_trace, retry_trace methods for agent learning.

    Workflow:
    - Use record_transition() after each env step to add (obs, action, reward, info) to buffer.
    - save_to_jsonl() writes current buffer to JSONL file for persistent replay or analysis.
    - load_from_jsonl() loads transitions from JSONL file, replacing buffer (for replay/debug).
    - episode_summary() computes stats (length, reward, actions) from buffer.
    - clear() empties buffer between episodes.
    - critique_trace() applies a critique_fn to episode trace, returns feedback.
    - retry_trace() applies a retry_fn to episode trace + critique, returns new trace.

    Buffer behavior:
    - Buffer is capped by max_transitions (default 1500). Oldest transitions are dropped only when buffer length exceeds max_transitions, keeping the newest.
    - Buffer is always in-memory and exposed via .transitions (read-only property).
    - Each transition is a dict: {observation, action, reward, info (optional)}.
    - JSONL persistence allows trace replay and debugging outside normal env loop.

    Usage:
        recorder = EpisodeRecorder(out_path="episode.jsonl")
        recorder.record_transition(obs, action, reward, info)
        recorder.save_to_jsonl()  # writes JSONL file
        recorder.load_from_jsonl("episode.jsonl")  # replay episode
        summary = recorder.episode_summary()

    Design notes:
        - Transition buffer is capped by max_transitions (default 1500) for memory efficiency.
        - Transitions are dicts with keys: observation, action, reward, info (optional).
        - Supports filtering transitions for analysis/debugging.
        - Self-reflection loop: critique and retry for learning.
    """
    def __init__(self, out_path: Optional[str] = None, max_transitions: Optional[int] = 1500):
        self.out_path = out_path
        self._transitions: List[Dict[str, Any]] = []
        self.max_transitions = max_transitions
        self._demonstrations: List[Dict[str, Any]] = []  # demonstration injection buffer
        self._last_critique: Optional[Any] = None  # stores last critique feedback
        self._last_retry_trace: Optional[List[Dict[str, Any]]] = None  # stores last retried trace

    @property
    def transitions(self) -> List[Dict[str, Any]]:
        """
        Access the current episode transitions buffer (read-only).
        """
        return self._transitions

    @property
    def demonstrations(self) -> List[Dict[str, Any]]:
        """
        Access the current demonstration buffer (read-only).
        """
        return self._demonstrations

    @property
    def last_critique(self) -> Optional[Any]:
        """
        Access the last critique feedback (read-only).
        """
        return self._last_critique

    @property
    def last_retry_trace(self) -> Optional[List[Dict[str, Any]]]:
        """
        Access the last retried trace (read-only).
        """
        return self._last_retry_trace

    def record_transition(self, observation: Any, action: Any, reward: float, info: Optional[Dict[str, Any]] = None):
        """
        Add a single transition to the episode buffer.
        Args:
            observation: Environment observation/state.
            action: Action taken by agent.
            reward: Reward obtained.
            info: Optional info dict from environment.
        """
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            transition["info"] = info
        self._transitions.append(transition)
        # Buffer cap: drop oldest if over max_transitions
        if self.max_transitions is not None and len(self._transitions) > self.max_transitions:
            self._transitions.pop(0)

    def clear(self):
        """
        Empty the transition buffer for the episode.
        """
        self._transitions = []

    def clear_demonstrations(self):
        """
        Empty the demonstration buffer.
        """
        self._demonstrations = []

    def inject_demonstration(self, observation: Any, action: Any, reward: float, info: Optional[Dict[str, Any]] = None):
        """
        Add a demonstration transition to the demonstration buffer.
        Args:
            observation: Example observation/state.
            action: Example action.
            reward: Example reward.
            info: Optional info dict.
        """
        demo = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            demo["info"] = info
        self._demonstrations.append(demo)

    def save_to_jsonl(self, path: Optional[str] = None):
        """
        Save buffered episode transitions to a JSONL file.
        Args:
            path: Optional override path to save file.
        """
        target_path = path or self.out_path
        if target_path is None:
            raise ValueError("No output path specified for episode recorder.")
        with open(target_path, 'w', encoding='utf-8') as f:
            for transition in self._transitions:
                json.dump(transition, f)
                f.write('\n')

    def load_from_jsonl(self, path: Optional[str] = None):
        """
        Load transitions from a JSONL file into the buffer. Overwrites current episode buffer.
        Args:
            path: Optional override path to load file.
        """
        source_path = path or self.out_path
        if source_path is None:
            raise ValueError("No input path specified for episode recorder.")
        transitions = []
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        transitions.append(json.loads(line))
                    except Exception:
                        continue
        self._transitions = transitions

    def episode_summary(self) -> Dict[str, Any]:
        """
        Compute summary stats for the current episode.
        Returns:
            dict with keys: length, total_reward, actions, unique_actions, reward_per_step
        """
        length = len(self._transitions)
        total_reward = sum(t.get("reward", 0) for t in self._transitions)
        actions = [t.get("action") for t in self._transitions]
        unique_actions = sorted(set(actions))
        reward_per_step = [t.get("reward", 0) for t in self._transitions]
        summary = {
            "length": length,
            "total_reward": total_reward,
            "actions": actions,
            "unique_actions": unique_actions,
            "reward_per_step": reward_per_step,
        }
        return summary

    def filter_transitions(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Return filtered transitions matching filter_fn.
        Args:
            filter_fn: function that returns True for transitions to keep.
        Returns:
            List of matching transitions.
        """
        return [t for t in self._transitions if filter_fn(t)]

    def save_to_csv(self, path: Optional[str] = None):
        """
        Save transitions to CSV file (flattened info fields).
        Args:
            path: Optional override path to save file.
        """
        target_path = path or (self.out_path and self.out_path.replace('.jsonl', '.csv'))
        if target_path is None:
            raise ValueError("No output path specified for CSV export.")
        rows = []
        for t in self._transitions:
            flat_t = {k: v for k, v in t.items() if k != "info"}
            if "info" in t:
                info = t["info"]
                if isinstance(info, dict):
                    for k, v in info.items():
                        flat_t[f'info.{k}'] = v
            rows.append(flat_t)
        # Collect all unique field names
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())
        fieldnames = sorted(fieldnames)
        with open(target_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # Self-reflection loop scaffold
    def critique_trace(self, critique_fn: Callable[[List[Dict[str, Any]]], Any]) -> Any:
        """
        Apply a critique_fn to the current episode trace and store feedback.
        Args:
            critique_fn: function that takes episode trace (list of transitions) and returns critique feedback.
        Returns:
            Critique feedback (arbitrary type).
        """
        feedback = critique_fn(self._transitions)
        self._last_critique = feedback
        return feedback

    def retry_trace(self, retry_fn: Callable[[List[Dict[str, Any]], Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Apply a retry_fn to the current episode trace and last critique feedback to produce a new trace.
        Args:
            retry_fn: function that takes episode trace and critique feedback, returns new episode trace (list of transitions).
        Returns:
            New episode trace (list of transitions).
        """
        new_trace = retry_fn(self._transitions, self._last_critique)
        self._last_retry_trace = new_trace
        return new_trace
