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
- [ ] TextWorld-style grid game wrapper
- [ ] Tic-tac-toe self-play
- [ ] Simple negotiation env

## Phase 4: learning
- [ ] In-context demonstration injection
- [ ] Self-reflection loop (trace -> critique -> retry)
- [ ] Reject-sampling fine-tune scaffold

## Phase 5: eval
- [ ] Win-rate vs. baselines
- [ ] Cost-per-episode tracking
- [ ] Side-by-side trace comparison
