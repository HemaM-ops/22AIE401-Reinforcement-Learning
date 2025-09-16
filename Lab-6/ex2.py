# mc_control_gridworld.py
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# --- Environment (same as Exercise 1) ---
class GridWorld5x5:
    def __init__(self, start=(0,0), goal=(4,4)):
        self.rows = 5
        self.cols = 5
        self.start = start
        self.goal = goal
        self.actions = [0,1,2,3]  # Up, Right, Down, Left
        self.delta = {
            0: (-1, 0),  # Up
            1: (0, +1),  # Right
            2: (+1, 0),  # Down
            3: (0, -1),  # Left
        }

    def step(self, state, action):
        if state == self.goal:
            return state, 0.0, True
        dr, dc = self.delta[action]
        nr = state[0] + dr
        nc = state[1] + dc
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            next_state = state
        else:
            next_state = (nr, nc)
        if next_state == self.goal:
            return next_state, 10.0, True
        else:
            return next_state, -1.0, False

    def is_terminal(self, s):
        return s == self.goal

# --- Epsilon-greedy policy utilities ---
def epsilon_greedy_policy(Q, env, epsilon):
    """Return a policy function that given a state returns an action."""
    def policy_fn(state):
        # choose greedy with prob 1-epsilon, else random
        if random.random() < (1 - epsilon):
            # find argmax Q(s,a)
            q_vals = [Q.get((state,a), 0.0) for a in env.actions]
            max_q = max(q_vals)
            # tie-breaking randomly among argmax
            candidates = [a for a, q in zip(env.actions, q_vals) if q == max_q]
            return random.choice(candidates)
        else:
            return random.choice(env.actions)
    return policy_fn

# --- Episode generation for given policy ---
def generate_episode(env, policy, start_state):
    episode = []
    s = start_state
    done = False
    while not done:
        a = policy(s)
        s_next, r, done = env.step(s, a)
        episode.append((s, a, r))
        s = s_next
    return episode

# --- Every-visit MC control (sample-average returns) ---
def mc_control_every_visit(env, num_episodes=100000, gamma=0.99, epsilon=0.1, seed=0, print_every=10000):
    random.seed(seed)
    np.random.seed(seed)

    # initialize
    Q = defaultdict(float)          # Q[(s,a)] = value
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    # initial policy: epsilon-greedy with current Q
    policy = epsilon_greedy_policy(Q, env, epsilon)

    episode_returns = []   # store total reward per episode
    running_avg_returns = []  # cumulative average (or sliding window)

    for ep in range(1, num_episodes+1):
        # generate episode under current policy
        episode = generate_episode(env, policy, start_state=env.start)
        # compute returns from end
        T = len(episode)
        rewards = [r for (_,_,r) in episode]

        # compute G for each time step (every-visit -> process all visits)
        # We'll process t from 0..T-1
        for t in range(T):
            s_t, a_t, _ = episode[t]
            # compute G_t
            G = 0.0
            pow_g = 1.0
            for k in range(t, T):
                G += pow_g * rewards[k]
                pow_g *= gamma
            # every-visit: accumulate for (s_t,a_t)
            returns_sum[(s_t,a_t)] += G
            returns_count[(s_t,a_t)] += 1
            Q[(s_t,a_t)] = returns_sum[(s_t,a_t)] / returns_count[(s_t,a_t)]

        # update policy to be epsilon-greedy wrt latest Q
        policy = epsilon_greedy_policy(Q, env, epsilon)

        # record episode return (total undiscounted reward for plotting)
        ep_return = sum(rewards)
        episode_returns.append(ep_return)

        if ep % print_every == 0:
            avg_recent = np.mean(episode_returns[-print_every:])
            print(f"Episode {ep}/{num_episodes}  avg_return(last {print_every}) = {avg_recent:.3f}")

    # prepare policy grid (pick greedy action for each state)
    policy_grid = np.full((env.rows, env.cols), '', dtype=object)
    arrow = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            if env.is_terminal(s):
                policy_grid[r,c] = 'G'
            else:
                q_vals = [Q.get((s,a), 0.0) for a in env.actions]
                max_q = max(q_vals)
                # tie-break randomly but deterministic for display: choose first argmax
                best_actions = [a for a, q in zip(env.actions, q_vals) if q == max_q]
                policy_grid[r,c] = arrow[random.choice(best_actions)]

    # compute cumulative average return for plotting
    cum_avg = np.cumsum(episode_returns) / np.arange(1, len(episode_returns)+1)

    return Q, policy_grid, episode_returns, cum_avg

# --- Plotting helpers ---
def plot_learning_curve(cum_avg, title="Learning curve (cumulative avg return)"):
    plt.figure(figsize=(8,4))
    plt.plot(cum_avg)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative average return")
    plt.title(title)
    plt.grid(True)
    plt.show()

def print_policy_grid(policy_grid, title="Learned policy"):
    print(title)
    for r in range(policy_grid.shape[0]):
        row = ' '.join([f"{cell:>2}" for cell in policy_grid[r,:]])
        print(row)

# --- Main runner ---
if __name__ == "__main__":
    env = GridWorld5x5(start=(0,0), goal=(4,4))
    Q, policy_grid, ep_returns, cum_avg = mc_control_every_visit(env,
                                                                 num_episodes=100000,
                                                                 gamma=0.99,
                                                                 epsilon=0.1,
                                                                 seed=123,
                                                                 print_every=20000)
    print("\nFinal greedy policy (G denotes goal):")
    print_policy_grid(policy_grid)
    plot_learning_curve(cum_avg, title="MC Control (every-visit) cumulative average return (100k ep)")
