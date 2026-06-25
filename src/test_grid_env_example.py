from grid_env import SimpleGridWorldEnv

if __name__ == "__main__":
    env = SimpleGridWorldEnv(grid_size=5, max_steps=10)
    obs, info = env.reset()
    print("Initial observation:")
    print(obs)
    print("Initial info:")
    print(info)
    total_reward = 0.0
    for step in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: action={env.ACTIONS[action]}, reward={reward}, done={done}")
        print("Observation:", obs)
        print("Info:", info)
        if done:
            print("Episode finished.")
            break
    print(f"Total reward: {total_reward}")
