# Trajectory and discount-based return calculation
trajectory = [
    {"state": (0, 0), "action": "right"},
    {"state": (0, 1), "action": "down"},
    {"state": (1, 1), "action": "right"},
    {"state": (1, 2), "action": "down"},
    {"state": (2, 2), "action": "deliver"},
]

def get_reward(action):
    if action == "deliver":
        return 5
    else:
        return -1

gamma = 0.8
rewards = []
Gt = 0
for i, step in enumerate(trajectory):
    r = get_reward(step["action"])
    rewards.append(r)
    Gt += (gamma ** i) * r
print(f"Rewards: {rewards}")
print(f"Cumulative Return G₁: {Gt:.3f}")