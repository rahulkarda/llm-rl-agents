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
    1. Instantiate an EpisodeRecorder (optionally with out_path):
        recorder = EpisodeRecorder(out_path="episode.jsonl")
    2. After each environment step, record the transition:
        recorder.record_transition(observation, action, reward, info)
    3. Save episode to JSONL for later replay or analysis:
        recorder.save_to_jsonl()  # Uses out_path
    4. Load and replay an episode:
        recorder.load_from_jsonl("episode.jsonl")
        for t in recorder.transitions:
            print(t)
    5. Compute summary statistics:
        summary = recorder.episode_summary()
        print(summary)
    6. Optionally, select transitions for fine-tuning:
        samples = recorder.reject_sample_for_finetune(threshold=0.8)
    7. Use critique_trace and retry_trace for self-reflection learning loop (scaffold).

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
        Add a demonstration transition to the demonstration buffer for in-context learning.
        """
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward
        }
        if info is not None:
            transition["info"] = info
        self._demonstrations.append(transition)

    def save_to_jsonl(self, out_path: Optional[str] = None):
        """
        Persist current episode transitions to JSONL file.
        Args:
            out_path: Optional override path. Uses self.out_path if not provided.
        """
        path = out_path or self.out_path
        assert path is not None, "No output path provided for JSONL save."
        with open(path, "w") as f:
            for t in self._transitions:
                f.write(json.dumps(t) + "\n")

    def load_from_jsonl(self, in_path: Optional[str] = None):
        """
        Load episode transitions from a JSONL file, replacing current buffer.
        Args:
            in_path: Optional override path. Uses self.out_path if not provided.
        """
        path = in_path or self.out_path
        assert path is not None, "No input path provided for JSONL load."
        transitions = []
        with open(path, "r") as f:
            for line in f:
                transitions.append(json.loads(line.strip()))
        self._transitions = transitions

    def episode_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for current episode trace.
        Returns:
            Dict with keys: length, total_reward, actions, reward_per_step
        """
        length = len(self._transitions)
        total_reward = sum(t["reward"] for t in self._transitions)
        actions = [t["action"] for t in self._transitions]
        reward_per_step = [t["reward"] for t in self._transitions]
        return {
            "length": length,
            "total_reward": total_reward,
            "actions": actions,
            "reward_per_step": reward_per_step
        }

    def export_to_csv(self, csv_path: str):
        """
        Export episode transitions to CSV for external analysis.
        Each row is a flattened dict.
        """
        if not self._transitions:
            return
        keys = set()
        flat_transitions = []
        for t in self._transitions:
            flat = flatten_dict(t)
            flat_transitions.append(flat)
            keys.update(flat.keys())
        keys = sorted(keys)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for t in flat_transitions:
                writer.writerow({k: t.get(k, "") for k in keys})

    def filter_transitions(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Return transitions satisfying filter_fn.
        Args:
            filter_fn: Callable that takes a transition dict and returns True/False.
        Returns:
            List of transitions.
        """
        return [t for t in self._transitions if filter_fn(t)]

    def critique_trace(self, critique_fn: Callable[[List[Dict[str, Any]]], Any]) -> Any:
        """
        Apply a critique function to the episode trace for self-reflection.
        Args:
            critique_fn: Callable taking episode trace (list of transitions), returns feedback.
        Returns:
            Feedback object (arbitrary type).
        """
        feedback = critique_fn(self._transitions)
        self._last_critique = feedback
        return feedback

    def retry_trace(self, retry_fn: Callable[[List[Dict[str, Any]], Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Retry episode trace given critique feedback.
        Args:
            retry_fn: Callable taking episode trace and critique, returns new trace.
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
