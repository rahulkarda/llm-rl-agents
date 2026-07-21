# Roadmap

## Phase 1: agent harness
- [x] Base `Agent` interface (observe, act)
- [x] Random and heuristic baselines
- [x] Episode recorder (state, action, reward sequence to JSONL)
- [x] Replay viewer

## Phase 2: LLM policies
- [x] Prompted-LLM policy on a text-observation env
- [x] Tool-use parsing (action JSON extraction with retries)
- [x] System prompt templating per env
- [x] Token + latency instrumentation

## Phase 3: environments
- [x] TextWorld-style grid game wrapper
- [x] Tic-tac-toe self-play
- [x] Simple negotiation env

## Phase 4: learning
- [x] In-context demonstration injection
- [x] Self-reflection loop (trace -> critique -> retry)
- [x] Reject-sampling fine-tune scaffold

## Phase 5: eval
- [x] Win-rate vs. baselines
- [x] Cost-per-episode tracking
- [x] Side-by-side trace comparison
- [x] Test for deep_get utility
