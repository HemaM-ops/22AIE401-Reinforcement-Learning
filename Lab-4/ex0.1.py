import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=True)  # stochastic transitions
obs, info = env.reset()
total_reward = 0
for t in range(1000):
    action = env.action_space.sample()  # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print (f"{action=},{reward=}, {obs=}, {terminated=}, {info=}")
    if terminated or truncated:
        break
print("Total reward:", total_reward)
env.close()
