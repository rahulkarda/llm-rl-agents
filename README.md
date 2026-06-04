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

## Design notes

- All agent classes expose a unified `act(observation)` method for compatibility.
- Recorder supports episode summary statistics and JSONL persistence for debugging and analysis.
- The project targets classic RL environments (e.g., CartPole, grid games) as well as text-based tasks.
