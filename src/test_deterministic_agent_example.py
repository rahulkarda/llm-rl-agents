import gymnasium as gym
from agent import DeterministicAgent

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # DeterministicAgent always returns action 0 by default
    agent = DeterministicAgent(env.action_space)
    obs, _ = env.reset()
    action = agent.act(obs)
    print(f"DeterministicAgent action: {action}")
    assert env.action_space.contains(action), f"Action {action} not in space"
    # Try setting fixed_action_index to 1 (for Discrete)
    agent.set_fixed_action_index(1)
    action2 = agent.act(obs)
    print(f"DeterministicAgent action (fixed_action_index=1): {action2}")
    assert env.action_space.contains(action2), f"Action {action2} not in space"
    print("DeterministicAgent example test passed.")
