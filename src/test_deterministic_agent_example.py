import gymnasium as gym
from agent import DeterministicAgent

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DeterministicAgent(env.action_space)  # Defaults to action 0 for Discrete
    obs, _ = env.reset()
    action = agent.act(obs)
    print(f"DeterministicAgent action (Discrete): {action}")
    assert env.action_space.contains(action), f"Action {action} not in space"

    # Test with fixed_action parameter
    agent2 = DeterministicAgent(env.action_space, fixed_action=1)
    action2 = agent2.act(obs)
    print(f"DeterministicAgent action (fixed_action=1): {action2}")
    assert env.action_space.contains(action2), f"Action {action2} not in space"

    print("DeterministicAgent example test passed.")
