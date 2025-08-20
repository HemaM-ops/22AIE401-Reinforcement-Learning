import numpy as np

# Gridworld parameters
n = 3  # 3x3 grid
gamma = 1.0  # discount factor
theta = 1e-4  # small threshold for convergence
actions = ['U', 'D', 'L', 'R']

# Initialize value function
V = np.zeros((n, n))

# Define transition function
def step(state, action):
    i, j = state
    if state == (2, 2):  # Goal state
        return state, 0
    
    if action == 'U':
        i = max(i - 1, 0)
    elif action == 'D':
        i = min(i + 1, n - 1)
    elif action == 'L':
        j = max(j - 1, 0)
    elif action == 'R':
        j = min(j + 1, n - 1)
    
    reward = -1
    if (i, j) == (2, 2):
        reward = 0
    return (i, j), reward

# Iterative policy evaluation
def policy_evaluation(V, gamma=1.0, theta=1e-4):
    while True:
        delta = 0
        new_V = V.copy()
        
        for i in range(n):
            for j in range(n):
                state = (i, j)
                if state == (2, 2):  # skip terminal state
                    continue
                
                value = 0
                for action in actions:
                    next_state, reward = step(state, action)
                    value += 0.25 * (reward + gamma * V[next_state])
                
                new_V[state] = value
                delta = max(delta, abs(new_V[state] - V[state]))
        
        V = new_V
        if delta < theta:
            break
    return V

# Run policy evaluation
V_final = policy_evaluation(V)
print("Final Value Function (Policy Evaluation):")
print(np.round(V_final, 2))