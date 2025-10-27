
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Exercise 2: SARSA(λ) Control on 3×3 Deterministic GridWorld
# Start: bottom-left (2,0), Goal: top-right (0,2)
# Step reward = -0.04, Goal reward = +1.0, γ = 0.9
# Actions: 0=Up, 1=Right, 2=Down, 3=Left
# ε-greedy exploration (ε=0.1)
# ------------------------------

class Grid3x3:
    def __init__(self, gamma=0.9):
        self.rows = 3
        self.cols = 3
        self.nS = self.rows*self.cols
        self.nA = 4
        self.start = (2,0)
        self.goal = (0,2)
        self.gamma = gamma

    def rc2s(self, r, c): return r*self.cols + c
    def s2rc(self, s): return divmod(s, self.cols)

    def reset(self):
        return self.rc2s(*self.start)

    def is_goal(self, s):
        return s == self.rc2s(*self.goal)

    def step(self, s, a):
        r, c = self.s2rc(s)
        if a == 0: r = max(0, r-1)            # Up
        elif a == 1: c = min(self.cols-1, c+1) # Right
        elif a == 2: r = min(self.rows-1, r+1) # Down
        elif a == 3: c = max(0, c-1)          # Left
        s_next = self.rc2s(r, c)
        if self.is_goal(s_next):
            return s_next, 1.0, True
        else:
            return s_next, -0.04, False

def epsilon_greedy(Q, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    q = Q[s]
    best = np.flatnonzero(q == np.max(q))
    return np.random.choice(best)

def sarsa_lambda(env, lam=0.8, alpha=0.1, epsilon=0.1, episodes=500, max_steps=200):
    Q = np.zeros((env.nS, env.nA))
    returns = []
    gamma = env.gamma

    for ep in range(episodes):
        e = np.zeros_like(Q)  # eligibility traces for (s,a)
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        done = False
        G = 0.0

        for t in range(max_steps):
            s_next, r, done = env.step(s, a)
            G += r
            if done:
                delta = r - Q[s, a]
                e[s, a] += 1.0
                Q += alpha * delta * e
                e *= gamma * lam
                returns.append(G)
                break
            else:
                a_next = epsilon_greedy(Q, s_next, epsilon)
                delta = r + gamma * Q[s_next, a_next] - Q[s, a]
                e[s, a] += 1.0
                Q += alpha * delta * e
                e *= gamma * lam
                s, a = s_next, a_next
        else:
            # max_steps reached
            returns.append(G)

    return Q, np.array(returns)

def print_greedy_policy(Q, env):
    arrows = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    grid = []
    for s in range(env.nS):
        if env.is_goal(s):
            grid.append('G')
        else:
            grid.append(arrows[np.argmax(Q[s])])
    for r in range(env.rows):
        print("  ".join(grid[r*env.cols:(r+1)*env.cols]))

if __name__ == "__main__":
    env = Grid3x3(gamma=0.9)
    lambdas = [0.0, 0.3, 0.6, 0.9]
    episodes = 500
    alpha = 0.1
    epsilon = 0.1

    all_returns = []
    plt.figure()
    for lam in lambdas:
        Q, returns = sarsa_lambda(env, lam=lam, alpha=alpha, epsilon=epsilon, episodes=episodes)
        # Simple moving average for smoother curve (window=10)
        if len(returns) >= 10:
            ma = np.convolve(returns, np.ones(10)/10, mode='valid')
            plt.plot(ma, label=f"λ={lam}")
        else:
            plt.plot(returns, label=f"λ={lam}")
        all_returns.append((lam, Q, returns))

    plt.xlabel("Episode")
    plt.ylabel("Average Return (moving avg window=10)")
    plt.title("SARSA(λ) on 3×3 GridWorld")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print final greedy policy for best-performing λ (by mean return of last 50 episodes)
    best_idx, best_score = 0, -1e9
    for i, (lam, Q, returns) in enumerate(all_returns):
        tail = returns[-50:] if len(returns) >= 50 else returns
        score = np.mean(tail) if len(tail) > 0 else -1e9
        if score > best_score:
            best_score = score
            best_idx = i
    lam_best, Q_best, _ = all_returns[best_idx]
    print(f"\nBest λ by late-episode return: {lam_best:.1f}")
    print("Greedy policy (rows top→bottom):")
    print_greedy_policy(Q_best, env)
