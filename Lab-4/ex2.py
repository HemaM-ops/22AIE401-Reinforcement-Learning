import gymnasium as gym

def run_frozenlake(is_slippery=True):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode=None)
    total_rewards = []
    
    for episode in range(50):
        state, info = env.reset(seed=42)
        done = False
        total_reward = 0
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    env.close()
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards)}\n")

print("=== Stochastic Environment (is_slippery=True) ===")
run_frozenlake(True)

print("=== Deterministic Environment (is_slippery=False) ===")
run_frozenlake(False)