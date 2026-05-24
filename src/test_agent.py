import gymnasium as gym
from agent import RandomAgent

class DummyObservation:
    pass

def test_random_agent_act():
    env = gym.make('CartPole-v1')
    agent = RandomAgent(env.action_space)
    obs = DummyObservation()
    action = agent.act(obs)
    assert env.action_space.contains(action), f"Action {action} not in space"

if __name__ == "__main__":
    test_random_agent_act()
    print("RandomAgent action sampling test passed.")
