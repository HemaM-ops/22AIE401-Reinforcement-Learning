import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")  # requires classic-control extras for rendering
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
