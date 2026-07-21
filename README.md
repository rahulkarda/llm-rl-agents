# llm-rl-agents

LLMs as policies in classic RL environments. Tool-using language agents that play games, solve puzzles, and learn through self-reflection. Sits at the intersection of LLM + RL.

## Goals

- Wrap small LLMs as Gym-compatible agents
- Compare prompted-only vs. fine-tuned policies on the same envs
- Build a tool-use harness so agents can call structured actions
- Trace and replay episodes for debugging

## Status

Bootstrapping. See [ROADMAP.md](ROADMAP.md).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

- See `src/agent.py` for agent interfaces and baselines (RandomAgent, DeterministicAgent).
- Use `EpisodeRecorder` in `src/recorder.py` to log episode transitions and replay traces.
- Tests for basic agent and recorder functionality are in `src/test_agent.py` and `src/test_recorder.py`.

### Agent interface and extension

Agents provide a unified interface for policy logic: observe the environment and emit an action.

**Core usage:**
```python
from agent import RandomAgent, DeterministicAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = RandomAgent(env.action_space)
observation, _ = env.reset()
action = agent.act(observation)
# Use action in env.step(action)
```

**Extension notes:**
- To make a new agent, subclass `Agent` and implement `act()`.
- For LLM-based policies, override `act()` to prompt the model and parse the action.
- `reset()` can be overridden to clear internal state between episodes.

### Grid world environment and heuristic agent example

This repo includes a simple text-based grid world environment for agent testing (`src/grid_env.py`). Agents can be evaluated in this environment, including heuristic baselines like `GreedyGridAgent`.

**Example usage:**
```python
from grid_env import SimpleGridWorldEnv
from agent import GreedyGridAgent

env = SimpleGridWorldEnv(grid_size=8, max_steps=100)
agent = GreedyGridAgent(env.action_space)
obs, info = env.reset()
done = False
while not done:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {info['step']}: {obs} | action={action} | reward={reward}")
```
- `GreedyGridAgent` moves toward the goal in the grid using simple heuristics.
- Observation is always a string for easy LLM prompting and compatibility.
- Action space: Discrete(4) (0=north, 1=south, 2=east, 3=west).

### EpisodeRecorder workflow

1. Instantiate an `EpisodeRecorder` (optionally with `out_path`):
   ```python
   from recorder import EpisodeRecorder
   recorder = EpisodeRecorder(out_path="episode.jsonl")
   ```
2. After each environment step, record the transition:
   ```python
   recorder.record_transition(observation, action, reward, info)
   ```
3. Save episode to JSONL for later replay or analysis:
   ```python
   recorder.save_to_jsonl()  # Uses out_path
   ```
4. Load and replay an episode:
   ```python
   recorder.load_from_jsonl("episode.jsonl")
   for t in recorder.transitions:
       print(t)
   ```
5. Compute summary statistics:
   ```python
   summary = recorder.episode_summary()
   print(summary)
   ```

**Reward summary utility:**

- Use `episode_reward_summary` from `src/utils.py` to summarize rewards across an episode trace.

Example usage:
```python
from utils import episode_reward_summary

# Suppose transitions is a list of dicts like:
transitions = [
    {"observation": "state1", "action": 0, "reward": 1.0},
    {"observation": "state2", "action": 1, "reward": 0.5},
    {"observation": "state3", "action": 2, "reward": -0.2},
]
summary = episode_reward_summary(transitions)
print(summary)  # {'total_reward': 1.3, 'mean_reward': 0.433, 'step_rewards': [1.0, 0.5, -0.2]}
```
- This utility also works with episode traces loaded by the recorder.

### LLM Agent integration

- Use `PromptedLLMAgent` from `src/llm_agent.py` to wrap a prompted OpenAI LLM as an RL agent.
- Requires an OpenAI API key (set environment variable `OPENAI_API_KEY`).

Example usage:
```python
from llm_agent import PromptedLLMAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = PromptedLLMAgent(env.action_space, system_prompt="You are playing CartPole.")
obs, _ = env.reset()
action = agent.act(obs)
# Use action in env.step(action)
```
- The agent prompts the LLM with the observation and expects a JSON action (e.g., `{"action": 0}`).
- If parsing fails or invalid action is returned, agent falls back to random action.
- Supports customizing system prompt and model (default: `gpt-3.5-turbo`).

### Utilities: deep_get for nested dict extraction

- Use `deep_get` from `src/utils.py` for robust extraction of values from nested dicts given a list of keys.

Example usage:
```python
from utils import deep_get

d = {'a': {'b': {'c': 5}}, 'x': {'y': 42}}
val1 = deep_get(d, ['a', 'b', 'c'])   # 5
val2 = deep_get(d, ['x', 'y'])        # 42
val3 = deep_get(d, ['a', 'b', 'z'])   # None
```
- `deep_get` returns the value at the nested path if present, or None if any key is missing along the path.
- It safely handles missing keys and non-dict intermediate values without raising errors.
- Example edge case:
```python
val4 = deep_get(d, ['a', 'b'])        # {'c': 5}
val5 = deep_get(d, ['a', 'b', 'c', 'd']) # None (since 'c' is not a dict)
```
- Useful for safely handling deeply nested info dicts or episode transitions.

## Design notes

- All agent classes expose a unified `act(observation)` method for compatibility.
- Recorder supports episode summary statistics and JSONL persistence for debugging and analysis.
- The project targets classic RL environments (e.g., CartPole, grid games) as well as text-based tasks.

