# Trajectory and discount-based return calculation
trajectory = [
    {"state": (0, 0), "action": "right"},
    {"state": (0, 1), "action": "down"},
    {"state": (1, 1), "action": "right"},
    {"state": (1, 2), "action": "down"},
    {"state": (2, 2), "action": "clean"},
]

def get_reward(action):
    if action == "clean":
        return 10
    elif action in {"right", "left", "up", "down"}:
        return -0.5
    else:
        return 0

gamma = 0.9
rewards = []
Gt = 0
for i, step in enumerate(trajectory):
    r = get_reward(step["action"])
    rewards.append(r)
    Gt += (gamma ** i) * r
print(f"Rewards: {rewards}")
print(f"Cumulative Return G₁: {Gt:.3f}")