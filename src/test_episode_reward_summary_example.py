from utils import episode_reward_summary

def test_episode_reward_summary_example():
    trace = [
        {'observation': 'foo', 'action': 1, 'reward': 0.5},
        {'observation': 'bar', 'action': 2, 'reward': 1.0},
        {'observation': 'baz', 'action': 0, 'reward': 0.0},
    ]
    summary = episode_reward_summary(trace)
    assert summary['total_reward'] == 1.5, f"Total reward incorrect: {summary['total_reward']}"
    assert abs(summary['mean_reward'] - 0.5) < 1e-8, f"Mean reward incorrect: {summary['mean_reward']}"
    assert summary['step_rewards'] == [0.5, 1.0, 0.0], f"Step rewards incorrect: {summary['step_rewards']}"
    print("episode_reward_summary example test passed.")

if __name__ == "__main__":
    test_episode_reward_summary_example()
