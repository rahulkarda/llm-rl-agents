import gymnasium as gym
from agent import DeterministicAgent

def test_deterministic_agent_discrete():
    env = gym.make('CartPole-v1')
    agent = DeterministicAgent(env.action_space)
    obs, _ = env.reset()
    action = agent.act(obs)
    assert env.action_space.contains(action), f"Action {action} not in space"
    assert action == 0, f"Default action for Discrete should be 0, got {action}"
    agent.set_fixed_action_index(1)
    action2 = agent.act(obs)
    assert action2 == 1, f"Fixed action index not set correctly: got {action2}"

def test_deterministic_agent_box():
    env = gym.make('Pendulum-v1')
    agent = DeterministicAgent(env.action_space)
    obs, _ = env.reset()
    action = agent.act(obs)
    low = env.action_space.low
    assert (action == low).all(), f"Box action should be action_space.low, got {action}"

def test_deterministic_agent_fixed_action():
    env = gym.make('CartPole-v1')
    agent = DeterministicAgent(env.action_space, fixed_action=1)
    obs, _ = env.reset()
    action = agent.act(obs)
    assert action == 1, f"fixed_action should be 1, got {action}"

if __name__ == "__main__":
    test_deterministic_agent_discrete()
    print("DeterministicAgent Discrete action test passed.")
    test_deterministic_agent_box()
    print("DeterministicAgent Box action test passed.")
    test_deterministic_agent_fixed_action()
    print("DeterministicAgent fixed_action test passed.")
