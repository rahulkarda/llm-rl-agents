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
        dict with keys: 'avg_cost', 'episode_costs', 'episodes'
    """
    episode_costs = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        truncated = False
        transitions = []
        agent.reset()  # Ensure agent state is reset before episode
        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            # Pack transition
            trans = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "info": info
            }
            transitions.append(trans)
        cost = compute_episode_cost(transitions, cost_keys=cost_keys)
        episode_costs.append(cost)
        env.close()
    avg_cost = sum(episode_costs) / len(episode_costs) if episode_costs else 0.0
    return {
        "avg_cost": avg_cost,
        "episode_costs": episode_costs,
        "episodes": episodes,
    }


def compare_traces_side_by_side(trace_a, trace_b, keys=None):
    """
    Compare two episode traces step-wise, showing differences for selected keys.
    Args:
        trace_a: List of transition dicts.
        trace_b: List of transition dicts.
        keys: List of keys to compare (default: ['observation', 'action', 'reward']).
    Returns:
        List of dicts with step-wise comparison.
    """
    if keys is None:
        keys = ["observation", "action", "reward"]
    max_steps = max(len(trace_a), len(trace_b))
    comparison = []
    for i in range(max_steps):
        a_dict = trace_a[i] if i < len(trace_a) else {}
        b_dict = trace_b[i] if i < len(trace_b) else {}
        a_vals = {k: a_dict.get(k, None) for k in keys}
        b_vals = {k: b_dict.get(k, None) for k in keys}
        diff = {k: (a_vals[k] != b_vals[k]) for k in keys}
        comparison.append({"step": i, "a": a_vals, "b": b_vals, "diff": diff})
    return comparison


def episode_reward_summary(trace):
    """
    Compute total, mean, and step-wise rewards from an episode trace.
    Args:
        trace: List of transition dicts (each with 'reward' key).
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
