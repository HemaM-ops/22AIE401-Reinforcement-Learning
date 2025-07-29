import random

# Grid and setup
grid_size = 3
num_cells = grid_size * grid_size
dirt = [0] * num_cells

# Randomly choose ~50% of tiles to be dirty
dirty_indices = random.sample(range(num_cells), k=4)
for idx in dirty_indices:
    dirt[idx] = 1

actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLEAN']
position = [0, 0]
max_steps = 15

# Helper functions
def pos_to_index(x, y):
    return x * grid_size + y

def move(pos, action):
    x, y = pos
    if action == 'UP' and x > 0:
        x -= 1
    elif action == 'DOWN' and x < grid_size - 1:
        x += 1
    elif action == 'LEFT' and y > 0:
        y -= 1
    elif action == 'RIGHT' and y < grid_size - 1:
        y += 1
    return [x, y]

def is_cleaned(dirt):
    return sum(dirt) == 0

# Simulation Loop
trajectory = []
gamma = 1.0
total_reward = 0

for step in range(max_steps):
    state = (position[0], position[1], tuple(dirt))
    action = random.choice(actions)
    
    if action == 'CLEAN':
        idx = pos_to_index(position[0], position[1])
        reward = 10 if dirt[idx] == 1 else -1
        dirt[idx] = 0
    else:
        reward = -1
        position = move(position, action)
    
    trajectory.append((state, action, reward))
    total_reward += reward
    
    if is_cleaned(dirt):
        break

# Compute Returns G_t
returns = []
G = 0
for _, _, reward in reversed(trajectory):
    G = reward + gamma * G
    returns.insert(0, G)

# Print Trajectory and Returns
print("\nTrajectory:")
for i, (s, a, r) in enumerate(trajectory):
    print(f"Step {i+1}: State={s}, Action={a}, Reward={r}, Gₜ={returns[i]}")

print(f"\nTotal Return: {returns[0]}")
