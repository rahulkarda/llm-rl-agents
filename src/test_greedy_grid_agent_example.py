from agent import GreedyGridAgent
import gymnasium as gym

class DummyActionSpace:
    def sample(self):
        return 0  # fallback action (should not be used in this example)
    n = 4  # Discrete(4)
    def contains(self, action):
        return 0 <= action < self.n

def make_obs(pos, goal):
    return f"You are at position ({pos[0]}, {pos[1]}). Goal is at ({goal[0]}, {goal[1]})."

if __name__ == "__main__":
    action_space = DummyActionSpace()
    agent = GreedyGridAgent(action_space)
    # Test moving east
    obs = make_obs((2, 2), (4, 2))
    action = agent.act(obs)
    print(f"Action (east expected=2): {action}")
    assert action == 2, "Agent should move east toward goal"
    # Test moving south
    obs = make_obs((2, 2), (2, 5))
    action = agent.act(obs)
    print(f"Action (south expected=1): {action}")
    assert action == 1, "Agent should move south toward goal"
    # Test moving west
    obs = make_obs((4, 2), (2, 2))
    action = agent.act(obs)
    print(f"Action (west expected=3): {action}")
    assert action == 3, "Agent should move west toward goal"
    # Test moving north
    obs = make_obs((2, 5), (2, 2))
    action = agent.act(obs)
    print(f"Action (north expected=0): {action}")
    assert action == 0, "Agent should move north toward goal"
    # Test random fallback when at goal
    obs = make_obs((2, 2), (2, 2))
    action = agent.act(obs)
    print(f"Action (random fallback): {action}")
    assert action_space.contains(action), "Fallback action should be valid"
    print("GreedyGridAgent heuristic test passed.")
