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
