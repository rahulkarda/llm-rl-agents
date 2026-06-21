import json
from typing import Any, Dict, Optional, List, Callable
import csv
from utils import flatten_dict

class EpisodeRecorder:
    """
    Stores episode transitions (observation, action, reward, info) for RL episodes.
    Supports saving transitions to JSONL for debugging/replay, loading episode traces, and computing summary statistics.
    Self-reflection loop scaffold: critique_trace, retry_trace methods for agent learning.
    Reject-sampling fine-tune scaffold: reject_sample_for_finetune method stub for learning integration.

    Workflow:
    - Use record_transition() after each env step to add (obs, action, reward, info) to buffer.
    - save_to_jsonl() writes current buffer to JSONL file for persistent replay or analysis.
    - load_from_jsonl() loads transitions from JSONL file, replacing buffer (for replay/debug).
    - episode_summary() computes stats (length, reward, actions) from buffer.
    - clear() empties buffer between episodes.
    - critique_trace() applies a critique_fn to episode trace, returns feedback.
    - retry_trace() applies a retry_fn to episode trace + critique, returns new trace.
    - reject_sample_for_finetune() performs reject-sampling selection for fine-tuning (scaffold).

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
        finetune_samples = recorder.reject_sample_for_finetune(threshold=0.8)

    Design notes:
        - Transition buffer is capped by max_transitions (default 1500) for memory efficiency.
        - Transitions are dicts with keys: observation, action, reward, info (optional).
        - Supports filtering transitions for analysis/debugging.
        - Self-reflection loop: critique and retry for learning.
        - Reject-sampling fine-tune scaffold: select transitions above threshold for fine-tuning.
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

    def save_to_jsonl(self, out_path: Optional[str] = None):
        """
        Save episode transitions to JSONL file.
        Args:
            out_path: Optional override for file path.
        """
        path = out_path or self.out_path
        if not path:
            raise ValueError("No out_path specified for JSONL export.")
        with open(path, "w", encoding="utf-8") as f:
            for t in self._transitions:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def load_from_jsonl(self, in_path: Optional[str] = None):
        """
        Load episode transitions from JSONL file, replacing current buffer.
        Args:
            in_path: Optional override for file path.
        """
        path = in_path or self.out_path
        if not path:
            raise ValueError("No in_path specified for JSONL import.")
        self._transitions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                self._transitions.append(t)

    def episode_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the current episode transitions.
        Returns:
            Dict with keys: length, total_reward, actions (list), info (flattened dict), etc.
        """
        length = len(self._transitions)
        total_reward = sum(t["reward"] for t in self._transitions)
        actions = [t["action"] for t in self._transitions]
        obs_first = self._transitions[0]["observation"] if self._transitions else None
        obs_last = self._transitions[-1]["observation"] if self._transitions else None
        info_flat = flatten_dict(self._transitions[-1]["info"]) if self._transitions and "info" in self._transitions[-1] else {}
        return {
            "length": length,
            "total_reward": total_reward,
            "actions": actions,
            "observation_first": obs_first,
            "observation_last": obs_last,
            "info_last_flat": info_flat,
        }

    def filter_transitions(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Return a filtered list of transitions matching filter_fn.
        Args:
            filter_fn: function that takes a transition dict, returns True/False.
        Returns:
            List of transitions passing filter_fn.
        """
        return [t for t in self._transitions if filter_fn(t)]

    def record_episode_csv(self, csv_path: str):
        """
        Save episode transitions to CSV for easy viewing.
        Args:
            csv_path: file path for CSV export.
        """
        if not self._transitions:
            return
        fieldnames = list(self._transitions[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self._transitions:
                writer.writerow(t)

    def filter_by_reward_threshold(self, threshold: float) -> List[Dict[str, Any]]:
        """
        Return transitions with reward >= threshold.
        Args:
            threshold: float reward threshold
        Returns:
            List of transitions.
        """
        return [t for t in self._transitions if t["reward"] >= threshold]

    def critique_trace(self, critique_fn: Callable[[List[Dict[str, Any]]], Any]) -> Any:
        """
        Apply a critique_fn to the current episode trace, returning feedback.
        Args:
            critique_fn: function that takes episode trace (list of transitions), returns feedback.
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

    def reject_sample_for_finetune(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Reject-sampling scaffold for fine-tuning.
        Select transitions with reward >= threshold for use in fine-tuning datasets.
        Args:
            threshold: float reward threshold (default 0.8)
        Returns:
            List of transitions suitable for fine-tuning.
        """
        # This is a stub/scaffold; more advanced selection logic can be added later.
        return [t for t in self._transitions if t["reward"] >= threshold]
