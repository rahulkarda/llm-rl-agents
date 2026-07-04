import gymnasium as gym
from grid_env import SimpleGridWorldEnv
from agent import RandomAgent
from eval import evaluate_win_rate

def test_grid_world_random_win_rate():
    """
    Evaluate RandomAgent win-rate on SimpleGridWorldEnv.
    Expect low win rate (random walk).
    """
    env_fn = lambda: SimpleGridWorldEnv(grid_size=8, max_steps=100)
    agent = RandomAgent(env_fn().action_space)
    stats = evaluate_win_rate(agent, env_fn, episodes=20)
    print("RandomAgent win rate on 8x8 grid:", stats["agent_win_rate"])
    # Win rate should be low, but nonzero
    assert 0.0 <= stats["agent_win_rate"] <= 0.5, f"Unexpected win rate: {stats['agent_win_rate']}"
    assert stats["episodes"] == 20
    print("Grid world win-rate eval test passed.")

if __name__ == "__main__":
    test_grid_world_random_win_rate()
