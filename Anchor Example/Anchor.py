
import numpy as np
import matplotlib.pyplot as plt

ARROWS = {0:'↑', 1:'→', 2:'↓', 3:'←'}

class Vacuum4x4:
    """
    4x4 grid. Deterministic moves with wall-clipping.
    Rewards:
      - step cost: -0.04
      - entering a dirty cell: additional -1.0
      - entering goal: +5.0 (terminal)
    """
    def __init__(self, dirty_cells=None, start=(3,0), goal=(0,3), step_cost=-0.04, dirty_penalty=-1.0, goal_reward=5.0, gamma=0.9):
        self.rows, self.cols = 4, 4
        self.nS = self.rows * self.cols
        self.nA = 4  # Up, Right, Down, Left
        self.start_rc = start
        self.goal_rc = goal
        self.step_cost = step_cost
        self.dirty_penalty = dirty_penalty
        self.goal_reward = goal_reward
        self.gamma = gamma
        if dirty_cells is None:
            dirty_cells = {(1,1), (2,2), (3,2)}
        self.dirty = set(dirty_cells)

    def rc2s(self, r, c): return r * self.cols + c
    def s2rc(self, s): return divmod(s, self.cols)
    def reset(self): return self.rc2s(*self.start_rc)
    def is_goal(self, s): return self.s2rc(s) == self.goal_rc

    def step(self, s, a):
        r, c = self.s2rc(s)
        if a == 0:   r = max(0, r-1)          # Up
        elif a == 1: c = min(self.cols-1, c+1) # Right
        elif a == 2: r = min(self.rows-1, r+1) # Down
        elif a == 3: c = max(0, c-1)          # Left
        s_next = self.rc2s(r, c)

        if self.is_goal(s_next):
            return s_next, self.goal_reward, True

        reward = self.step_cost
        if (r, c) in self.dirty:
            reward += self.dirty_penalty
        return s_next, reward, False

def epsilon_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    q = Q[s]
    best = np.flatnonzero(q == np.max(q))
    return np.random.choice(best)

def render_policy_grid(Q, env: Vacuum4x4, title="Smart Vacuum 4×4 — Dyna-Q Greedy Policy"):
    grid = []
    for s in range(env.nS):
        r, c = env.s2rc(s)
        if (r, c) == env.start_rc:
            cell = 'S'
        elif (r, c) == env.goal_rc:
            cell = 'G'
        elif (r, c) in env.dirty:
            cell = 'D'
        else:
            cell = ARROWS[np.argmax(Q[s])]
        grid.append(cell)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim(0, env.cols); ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(0, env.cols+1, 1)); ax.set_yticks(np.arange(0, env.rows+1, 1))
    ax.grid(True)

    for r in range(env.rows):
        for c in range(env.cols):
            val = grid[r*env.cols + c]
            ax.text(c+0.5, env.rows-1-r+0.5, val, ha='center', va='center', fontsize=16)
            if (r, c) in env.dirty:
                ax.add_patch(plt.Rectangle((c, env.rows-1-r), 1, 1, fill=False))

    ax.set_title(title)
    plt.show()

def q_learning(env: Vacuum4x4, episodes=300, alpha=0.1, eps=0.1, gamma=None, max_steps=200):
    if gamma is None: gamma = env.gamma
    Q = np.zeros((env.nS, env.nA))
    returns = []
    for ep in range(episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            s_next, r, done = env.step(s, a)
            G += r
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) * (not done) - Q[s, a])
            s = s_next
            if done:
                break
        returns.append(G)
    return Q, np.array(returns)

def dyna_q(env: Vacuum4x4, episodes=300, alpha=0.1, eps=0.1, gamma=None, planning_steps=10, max_steps=200, seed=0):
    if gamma is None: gamma = env.gamma
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.nS, env.nA))
    returns = []
    model_r = {}
    model_snext = {}
    observed_pairs = []

    for ep in range(episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            s_next, r, done = env.step(s, a)
            G += r
            # Real update
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) * (not done) - Q[s, a])
            # Model update
            model_r[(s, a)] = r
            model_snext[(s, a)] = s_next
            if (s, a) not in observed_pairs:
                observed_pairs.append((s, a))
            # Planning updates
            for _ in range(planning_steps):
                sp, ap = observed_pairs[rng.integers(len(observed_pairs))]
                rp = model_r[(sp, ap)]
                snp = model_snext[(sp, ap)]
                terminal = env.is_goal(snp)
                Q[sp, ap] += alpha * (rp + gamma * np.max(Q[snp]) * (not terminal) - Q[sp, ap])
            s = s_next
            if done:
                break
        returns.append(G)
    return Q, np.array(returns)

def moving_avg(x, k=10):
    if len(x) < k: return x
    return np.convolve(x, np.ones(k)/k, mode='valid')

def run_demo():
    env = Vacuum4x4(dirty_cells={(1,1), (2,2), (3,2)}, start=(3,0), goal=(0,3),
                    step_cost=-0.04, dirty_penalty=-1.0, goal_reward=5.0, gamma=0.9)
    episodes = 300
    alpha = 0.1
    eps = 0.1

    Q_q, ret_q = q_learning(env, episodes=episodes, alpha=alpha, eps=eps)
    Q_d, ret_d = dyna_q(env, episodes=episodes, alpha=alpha, eps=eps, planning_steps=10)

    # Plot learning curves
    plt.figure()
    plt.plot(moving_avg(ret_q, 10), label="Q-Learning (n=0)")
    plt.plot(moving_avg(ret_d, 10), label="Dyna-Q (n=10)")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (moving avg=10)")
    plt.title("4×4 Smart Vacuum: Q-Learning vs Dyna-Q")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Render greedy policy for Dyna-Q
    render_policy_grid(Q_d, env, title="Smart Vacuum 4×4 — Dyna-Q Greedy Policy")

if __name__ == "__main__":
    run_demo()
