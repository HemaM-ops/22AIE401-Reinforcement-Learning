import gymnasium as gym

# Create the environment
env = gym.make("CartPole-v1", render_mode=None)
state, info = env.reset(seed=42)

for step in range(500):
    action = env.action_space.sample()  # Random action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step: {step+1}, Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
    
    if terminated or truncated:
        state, info = env.reset()

env.close()