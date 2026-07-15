from utils import episode_reward_summary

if __name__ == "__main__":
    transitions = [
        {"observation": "state1", "action": 0, "reward": 1.0},
        {"observation": "state2", "action": 1, "reward": 0.5},
        {"observation": "state3", "action": 2, "reward": -0.2},
    ]
    summary = episode_reward_summary(transitions)
    print("transitions:", transitions)
    print("episode_reward_summary:", summary)  # Expected: {'total_reward': 1.3, 'mean_reward': 0.433..., 'step_rewards': [1.0, 0.5, -0.2]}
