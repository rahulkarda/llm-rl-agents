import gymnasium as gym
from agent import DeterministicAgent

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DeterministicAgent(env.action_space)
    obs, _ = env.reset()
    action = agent.act(obs)
    print(f"DeterministicAgent (default) action: {action}")
    assert env.action_space.contains(action), f"Action {action} not in space"

    # Test fixed_action
    agent2 = DeterministicAgent(env.action_space, fixed_action=1)
    action2 = agent2.act(obs)
    print(f"DeterministicAgent (fixed_action=1) action: {action2}")
    assert action2 == 1
    assert env.action_space.contains(action2)

    print("DeterministicAgent example test passed.")
