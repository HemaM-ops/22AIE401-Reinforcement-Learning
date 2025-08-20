import numpy as np

# Gridworld parameters
n = 3  # 3x3 grid
gamma = 1.0
theta = 1e-4
actions = ['U', 'D', 'L', 'R']
action_idx = {a: i for i, a in enumerate(actions)}

# Transition function
def step(state, action):
    i, j = state
    if state == (2, 2):  # goal
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

# Policy Iteration
def policy_iteration(gamma=1.0, theta=1e-4):
    # Initialize random policy: each action equal prob.
    policy = np.ones((n, n, len(actions))) / len(actions)
    V = np.zeros((n, n))

    while True:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            new_V = V.copy()
            for i in range(n):
                for j in range(n):
                    state = (i, j)
                    if state == (2, 2):  # skip terminal
                        continue
                    value = 0
                    for a, action in enumerate(actions):
                        next_state, reward = step(state, action)
                        value += policy[i, j, a] * (reward + gamma * V[next_state])
                    new_V[state] = value
                    delta = max(delta, abs(new_V[state] - V[state]))
            V = new_V
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for i in range(n):
            for j in range(n):
                state = (i, j)
                if state == (2, 2):
                    continue
                
                # old action
                old_action = np.argmax(policy[i, j])
                
                # compute action-values
                q_values = []
                for action in actions:
                    next_state, reward = step(state, action)
                    q_values.append(reward + gamma * V[next_state])
                
                # greedy action
                best_action = np.argmax(q_values)
                
                # update policy
                policy[i, j] = np.eye(len(actions))[best_action]
                
                if old_action != best_action:
                    policy_stable = False
        
        if policy_stable:
            return policy, V

# Run Policy Iteration
policy, V_opt = policy_iteration()
print("Optimal Value Function:")
print(np.round(V_opt, 2))

print("\nOptimal Policy (arrows):")
arrow_map = {'U':'↑','D':'↓','L':'←','R':'→'}
for i in range(n):
    row = ""
    for j in range(n):
        if (i,j) == (2,2):
            row += " G "
        else:
            row += " " + arrow_map[actions[np.argmax(policy[i,j])]] + " "
    print(row)
