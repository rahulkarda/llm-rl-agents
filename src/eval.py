"""
Evaluation utilities for RL agents: win-rate and cost tracking.

Functions:
- evaluate_win_rate(agent, env_fn, episodes, baseline, win_criteria): Evaluate agent vs baseline win-rate on an env, using custom win criteria.
- evaluate_cost_per_episode(agent, env_fn, episodes, cost_keys): Track average and per-episode cost (e.g., API usage) from episode traces.
- compare_traces_side_by_side(trace_a, trace_b, keys=None): Compare two episode traces step-wise, showing differences for selected keys.
- episode_reward_summary(trace): Compute total, mean, and step-wise rewards from an episode trace.

Workflow:
- Provide agent and env factory (env_fn: lambda returning env instance).
- Optionally provide baseline agent and win_criteria (function taking info dict, returns bool).
- For cost tracking, agent must propagate cost fields in info dicts during episode.
- For trace comparison, provide two traces (lists of transition dicts).
- For reward summary, provide a trace (list of dicts with 'reward' key).

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
    # Compute reward summary
    reward_summary = episode_reward_summary(trace_a)
    print(reward_summary)

Notes:
- evaluate_win_rate alternates agent and baseline (if provided), else always agent.
- win_criteria defaults to total_reward > 0.99 unless provided.
- evaluate_cost_per_episode expects cost keys (default ("cost", "api_cost")) in info dicts.
- Both functions return dicts with summary stats for downstream analysis.
- compare_traces_side_by_side returns a list of dicts, each showing step-wise comparison for supplied keys.
- episode_reward_summary returns dict with total, mean, and step-wise rewards.
"""
import gymnasium as gym
from agent import RandomAgent
from utils import compute_episode_cost, flatten_dict_keys, dict_values_to_list, episode_reward_summary


def evaluate_win_rate(agent, env_fn, episodes=100, baseline=None, win_criteria=None):
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
    agent_episodes = 0
    baseline_episodes = 0
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        info = {}
        # Use agent for even, baseline for odd episodes
        current_agent = agent if (baseline is None or ep % 2 == 0) else baseline
        current_agent.reset()  # Ensure agent state is reset before episode
        while not (done or truncated):
            action = current_agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(current_agent, 'step'):
                current_agent.step()
            total_reward += reward
        # Determine win
        if win_criteria:
            won = win_criteria(info)
        else:
            # Default: win if reward > 0.99 or env-specific
            won = total_reward > 0.99
        if current_agent is agent:
            agent_wins += int(won)
            agent_episodes += 1
        else:
            baseline_wins += int(won)
            baseline_episodes += 1
        env.close()
    agent_win_rate = agent_wins / agent_episodes if agent_episodes > 0 else 0.0
    baseline_win_rate = baseline_wins / baseline_episodes if baseline and baseline_episodes > 0 else None
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
        dict with keys: 'avg_cost', 'per_episode_costs', 'cost_keys'
    """
    per_episode_costs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_trace = []
        agent.reset()
        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            transition = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "info": info,
            }
            episode_trace.append(transition)
            if hasattr(agent, 'step'):
                agent.step()
        # Compute episode cost
        cost = compute_episode_cost(episode_trace, cost_keys=cost_keys)
        per_episode_costs.append(cost)
        env.close()
    avg_cost = sum(per_episode_costs) / len(per_episode_costs) if per_episode_costs else 0.0
    return {
        "avg_cost": avg_cost,
        "per_episode_costs": per_episode_costs,
        "cost_keys": cost_keys,
    }


def compare_traces_side_by_side(trace_a, trace_b, keys=None):
    """
    Compare two episode traces step-wise, showing differences for selected keys.
    Args:
        trace_a: List of transition dicts.
        trace_b: Same length, list of transition dicts.
        keys: List of keys to compare (optional; if None, uses all keys from both traces).
    Returns:
        List of dicts, each showing comparison for the step.
    """
    comparison = []
    len_a, len_b = len(trace_a), len(trace_b)
    max_len = max(len_a, len_b)
    # Determine keys
    if keys is None:
        keys = set()
        if trace_a:
            keys.update(flatten_dict_keys(trace_a[0]))
        if trace_b:
            keys.update(flatten_dict_keys(trace_b[0]))
        keys = sorted(list(keys))
    for i in range(max_len):
        step = {"step": i}
        a = trace_a[i] if i < len_a else None
        b = trace_b[i] if i < len_b else None
        for k in keys:
            step[f"a_{k}"] = a[k] if a and k in a else None
            step[f"b_{k}"] = b[k] if b and k in b else None
        comparison.append(step)
    return comparison


def episode_reward_summary(trace):
    """
    Compute total, mean, and step-wise rewards from an episode trace.
    Args:
        trace: list of dicts, each must have 'reward' key (can be missing or zero for a step).
    Returns:
        dict with keys: 'total_reward', 'mean_reward', 'step_rewards'
    """
    step_rewards = []
    for t in trace:
        # Try t['reward'], else t.get('reward', 0.0)
        r = t.get('reward', 0.0)
        step_rewards.append(r)
    total_reward = sum(step_rewards)
    mean_reward = total_reward / len(step_rewards) if step_rewards else 0.0
    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "step_rewards": step_rewards
    }
