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

## Design notes

- All agent classes expose a unified `act(observation)` method for compatibility.
- Recorder supports episode summary statistics and JSONL persistence for debugging and analysis.
- The project targets classic RL environments (e.g., CartPole, grid games) as well as text-based tasks.
