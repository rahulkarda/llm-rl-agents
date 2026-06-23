import gymnasium as gym
from agent import RandomAgent

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
        current_agent.reset()
        while not done:
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

if __name__ == "__main__":
    # Simple test: RandomAgent vs itself on CartPole
    env_fn = lambda: gym.make('CartPole-v1')
    agent = RandomAgent(env_fn().action_space)
    stats = evaluate_win_rate(agent, env_fn, episodes=10)
    print("Agent win rate:", stats["agent_win_rate"], "Wins:", stats["agent_wins"], "/", stats["episodes"])
