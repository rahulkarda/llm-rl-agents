"""
Evaluation utilities for RL agents: win-rate and cost tracking.

Functions:
- evaluate_win_rate(agent, env_fn, episodes, baseline, win_criteria): Evaluate agent vs baseline win-rate on an env, using custom win criteria.
- evaluate_cost_per_episode(agent, env_fn, episodes, cost_keys): Track average and per-episode cost (e.g., API usage) from episode traces.
- compare_traces_side_by_side(trace_a, trace_b, keys=None): Compare two episode traces step-wise, showing differences for selected keys.

Workflow:
- Provide agent and env factory (env_fn: lambda returning env instance).
- Optionally provide baseline agent and win_criteria (function taking info dict, returns bool).
- For cost tracking, agent must propagate cost fields in info dicts during episode.
- For trace comparison, provide two traces (lists of transition dicts).

Example usage:
    import gymnasium as gym
    from agent import RandomAgent
    env_fn = lambda: gym.make('CartPole-v1')
    agent = RandomAgent(env_fn().action_space)
    stats = evaluate_win_rate(agent, env_fn, episodes=10)
    print("Agent win rate:", stats["agent_win_rate"])
    cost_stats = evaluate_cost_per_episode(agent, env_fn, episodes=5)
    print("Average episode cost:", cost_stats["avg_cost"])
    # Compare two traces
    trace_a = [{"observation": "a", "action": 1, "reward": 0.5}]
    trace_b = [{"observation": "b", "action": 2, "reward": 0.7}]
    comparison = compare_traces_side_by_side(trace_a, trace_b)
    print(comparison)

Notes:
- evaluate_win_rate alternates agent and baseline (if provided), else always agent.
- win_criteria defaults to total_reward > 0.99 unless provided.
- evaluate_cost_per_episode expects cost keys (default ("cost", "api_cost")) in info dicts.
- Both functions return dicts with summary stats for downstream analysis.
- compare_traces_side_by_side returns a list of dicts, each showing step-wise comparison for supplied keys.
"""
import gymnasium as gym
from agent import RandomAgent
from utils import compute_episode_cost, flatten_dict_keys, dict_values_to_list

def evaluate_win_rate(agent, env_fn, episodes=50, baseline=None, win_criteria=None):
    """
    Evaluate win-rate of agent vs baseline on env_fn across episodes.
    Args:
        agent: Agent instance (must implement act()).
        env_fn: Callable that returns a new env instance (e.g., lambda: gym.make('CartPole-v1')).
        episodes: Number of episodes to run.
        baseline: Baseline agent (optional). If provided, alternates agent vs baseline.
        win_criteria: Callable (info dict -> bool) to determine win; if None, uses reward > 0.99 or env-specific.
    Returns:
        dict with keys: 'agent_win_rate', 'baseline_win_rate', 'agent_wins', 'baseline_wins', 'episodes'
    """
    agent_wins = 0
    baseline_wins = 0
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        info = {}
        # Use agent for even, baseline for odd episodes
        current_agent = agent if (baseline is None or ep % 2 == 0) else baseline
        current_agent.reset()  # Ensure agent state is reset before episode
        while not done:
            action = current_agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            done = done or truncated  # fix: episode ends if either is True
        # Determine win
        if win_criteria:
            won = win_criteria(info)
        else:
            # Default: win if reward > 0.99 or env-specific
            won = total_reward > 0.99
        if current_agent is agent:
            agent_wins += int(won)
        else:
            baseline_wins += int(won)
        env.close()
    agent_win_rate = agent_wins / episodes
    baseline_win_rate = baseline_wins / episodes if baseline else None
    return {
        "agent_win_rate": agent_win_rate,
        "baseline_win_rate": baseline_win_rate,
        "agent_wins": agent_wins,
        "baseline_wins": baseline_wins,
        "episodes": episodes,
    }

def evaluate_cost_per_episode(agent, env_fn, episodes=10, cost_keys=("cost", "api_cost")):
    """
    Evaluate agent cost per episode for episodes on env_fn.
    Args:
        agent: Agent instance.
        env_fn: Callable that returns a new env instance.
        episodes: Number of episodes.
        cost_keys: Tuple of keys to look for in transition dicts (top-level or under 'info').
    Returns:
        dict with keys: 'avg_cost', 'episode_costs', 'episodes'
    """
    episode_costs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        transitions = []
        agent.reset()  # Ensure agent state is reset before episode
        while not done:
            action = agent.act(obs)
            obs2, reward, done, truncated, info = env.step(action)
            done = done or truncated  # fix: episode ends if either is True
            transition = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "info": info,
            }
            # Optionally propagate cost if present in info or elsewhere
            for key in cost_keys:
                if key in info and isinstance(info[key], (int, float)):
                    transition[key] = info[key]
            transitions.append(transition)
            obs = obs2
        total_cost = compute_episode_cost(transitions, cost_keys=cost_keys)
        episode_costs.append(total_cost)
        env.close()
    avg_cost = sum(episode_costs) / len(episode_costs) if episode_costs else 0.0
    return {
        "avg_cost": avg_cost,
        "episode_costs": episode_costs,
        "episodes": episodes,
    }

def compare_traces_side_by_side(trace_a, trace_b, keys=None):
    """
    Compare two episode traces side-by-side step-wise.
    Args:
        trace_a: List of transition dicts (episode A).
        trace_b: List of transition dicts (episode B).
        keys: List of keys to compare (default: intersection of flat keys in both traces).
    Returns:
        List of dicts for each step: {'step': i, 'a': {...}, 'b': {...}, 'diff': {...}}
    """
    max_len = max(len(trace_a), len(trace_b))
    # Determine keys to compare
    if keys is None:
        keys_a = set(flatten_dict_keys(trace_a[0])) if trace_a else set()
        keys_b = set(flatten_dict_keys(trace_b[0])) if trace_b else set()
        keys = sorted(list(keys_a & keys_b)) if keys_a and keys_b else None
    comparison = []
    for i in range(max_len):
        ta = trace_a[i] if i < len(trace_a) else {}
        tb = trace_b[i] if i < len(trace_b) else {}
        # Extract values for keys
        if keys:
            vals_a = {k: ta.get(k, None) for k in keys}
            vals_b = {k: tb.get(k, None) for k in keys}
        else:
            # Use all keys
            all_keys = sorted(set(ta.keys()) | set(tb.keys()))
            vals_a = {k: ta.get(k, None) for k in all_keys}
            vals_b = {k: tb.get(k, None) for k in all_keys}
        # Diff: show keys where values differ
        diff = {k: (vals_a[k], vals_b[k]) for k in vals_a if vals_a[k] != vals_b[k]}
        comparison.append({
            'step': i,
            'a': vals_a,
            'b': vals_b,
            'diff': diff,
        })
    return comparison

if __name__ == "__main__":
    # Simple test: RandomAgent vs itself on CartPole
    env_fn = lambda: gym.make('CartPole-v1')
    agent = RandomAgent(env_fn().action_space)
    stats = evaluate_win_rate(agent, env_fn, episodes=10)
    print("Agent win rate:", stats["agent_win_rate"], "Wins:", stats["agent_wins"], "/", stats["episodes"])
    # Test cost-per-episode (CartPole does not have cost info, so all zero)
    cost_stats = evaluate_cost_per_episode(agent, env_fn, episodes=5)
    print("Average episode cost:", cost_stats["avg_cost"], "Episode costs:", cost_stats["episode_costs"])
    # Trace comparison example
    trace_a = [
        {"observation": "state1", "action": 0, "reward": 1.0},
        {"observation": "state2", "action": 1, "reward": 0.0},
    ]
    trace_b = [
        {"observation": "state1", "action": 0, "reward": 0.5},
        {"observation": "state2", "action": 2, "reward": 1.0},
    ]
    comparison = compare_traces_side_by_side(trace_a, trace_b, keys=["observation", "action", "reward"])
    for step_cmp in comparison:
        print(f"Step {step_cmp['step']}:")
        print("  A:", step_cmp['a'])
        print("  B:", step_cmp['b'])
        print("  Diff:", step_cmp['diff'])
