import random

grid = [
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'H'],
    ['F', 'F', 'F', 'H'],
    ['H', 'F', 'F', 'G']
]

grid_size = 4
start = [0, 0]
goal = [3, 3]
holes = [(1,1), (1,3), (2,3), (3,0)]

actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
action_to_delta = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Perpendicular mapping for stochastic slips
perpendicular = {
    'UP': ['LEFT', 'RIGHT'],
    'DOWN': ['LEFT', 'RIGHT'],
    'LEFT': ['UP', 'DOWN'],
    'RIGHT': ['UP', 'DOWN']
}

position = start.copy()
trajectory = []
gamma = 1.0
max_steps = 20
done = False

def move(pos, action):
    # Simulate stochastic behavior
    p = random.random()
    if p < 0.8:
        chosen_action = action
    elif p < 0.9:
        chosen_action = perpendicular[action][0]
    else:
        chosen_action = perpendicular[action][1]

    dx, dy = action_to_delta[chosen_action]
    new_x = max(0, min(grid_size - 1, pos[0] + dx))
    new_y = max(0, min(grid_size - 1, pos[1] + dy))
    return [new_x, new_y], chosen_action

# Simulate episode
for step in range(max_steps):
    if done:
        break

    state = tuple(position)
    intended_action = random.choice(actions)
    position, actual_action = move(position, intended_action)

    if tuple(position) in holes:
        reward = -10
        done = True
    elif position == goal:
        reward = 10
        done = True
    else:
        reward = -1

    trajectory.append((state, intended_action, actual_action, reward))

# Compute returns
returns = []
G = 0
for _, _, _, reward in reversed(trajectory):
    G = reward + gamma * G
    returns.insert(0, G)

# Display trajectory
print("\nTrajectory:")
for i, (s, intended, actual, r) in enumerate(trajectory):
    print(f"Step {i+1}: State={s}, Intended={intended}, Actual={actual}, Reward={r}, Gₜ={returns[i]}")
print(f"\nTotal Return: {returns[0]}")
