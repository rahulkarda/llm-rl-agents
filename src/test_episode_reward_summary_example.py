from utils import episode_reward_summary


def test_episode_reward_summary():
    transitions = [
        {"observation": "state1", "action": 0, "reward": 1.0},
        {"observation": "state2", "action": 1, "reward": 0.5},
        {"observation": "state3", "action": 2, "reward": -0.2},
    ]
    summary = episode_reward_summary(transitions)
    assert isinstance(summary, dict), "Summary should be a dict"
    assert "total_reward" in summary
    assert "mean_reward" in summary
    assert "step_rewards" in summary
    assert summary["total_reward"] == 1.3, f"Expected total_reward 1.3, got {summary['total_reward']}"
    assert summary["mean_reward"] == 1.3/3, f"Expected mean_reward {1.3/3}, got {summary['mean_reward']}"
    assert summary["step_rewards"] == [1.0, 0.5, -0.2], f"Expected step_rewards [1.0, 0.5, -0.2], got {summary['step_rewards']}"
    print("episode_reward_summary test passed.")

if __name__ == "__main__":
    test_episode_reward_summary()
