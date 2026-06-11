from grid_env import SimpleGridWorldEnv

def test_simple_grid_env_basic():
    env = SimpleGridWorldEnv(grid_size=4, max_steps=10)
    obs, _ = env.reset()
    assert isinstance(obs, str), f"Observation should be string, got: {type(obs)}"
    action = 2  # east
    obs2, reward, done, truncated, info = env.step(action)
    assert isinstance(obs2, str), "Returned observation should be string"
    assert reward in [0.0, 1.0], f"Reward should be 0.0 or 1.0, got: {reward}"
    assert isinstance(done, bool), f"Done should be bool, got: {type(done)}"
    assert isinstance(info, dict), f"Info should be dict, got: {type(info)}"

if __name__ == "__main__":
    test_simple_grid_env_basic()
    print("SimpleGridWorldEnv basic test passed.")
