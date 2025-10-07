
import numpy as np
import matplotlib.pyplot as plt

# Mini CliffWorld (4x8) for SARSA control
class MiniCliffWorld:
    def __init__(self, rows=4, cols=8, gamma=1.0, cliff_penalty=-100.0, step_reward=-1.0):
        self.rows, self.cols = rows, cols
        self.nS, self.nA = rows*cols, 4
        self.gamma, self.cliff_penalty, self.step_reward = gamma, cliff_penalty, step_reward
        self.start, self.goal = (rows-1, 0), (rows-1, cols-1)

    def s2rc(self, s): return divmod(s, self.cols)
    def rc2s(self, r, c): return r*self.cols + c
    def reset(self): return self.rc2s(*self.start)
    def is_goal(self, s): return s == self.rc2s(*self.goal)
    def is_cliff_rc(self, r, c): return r == self.rows-1 and 1 <= c <= self.cols-2

    def step(self, s, a):
        if self.is_goal(s): return s, 0.0, True
        r, c = self.s2rc(s)
        if a == 0: r = max(0, r-1)
        elif a == 1: c = min(self.cols-1, c+1)
        elif a == 2: r = min(self.rows-1, r+1)
        elif a == 3: c = max(0, c-1)
        if self.is_cliff_rc(r, c):
            return self.reset(), self.cliff_penalty, False
        s_next = self.rc2s(r, c)
        reward = 0.0 if self.is_goal(s_next) else self.step_reward
        done = self.is_goal(s_next)
        return s_next, reward, done

def epsilon_greedy(Q, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    q = Q[s]; maxq = np.max(q)
    best = np.flatnonzero(np.isclose(q, maxq))
    return np.random.choice(best)

def sarsa(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=500):
    Q = np.zeros((env.nS, env.nA))
    returns = []
    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        done, G, steps = False, 0.0, 0
        while not done and steps < 1000:
            s_next, r, done = env.step(s, a)
            G += r
            td_target = r if done else r + gamma * Q[s_next, epsilon_greedy(Q, s_next, epsilon)]
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, a = (s_next, epsilon_greedy(Q, s_next, epsilon)) if not done else (s_next, 0)
            steps += 1
        returns.append(G)
    return Q, returns

if __name__ == "__main__":
    env = MiniCliffWorld(rows=4, cols=8, gamma=1.0)
    Q, returns = sarsa(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=500)

    print("Final greedy policy (arrows; G=goal):")
    arrows = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    grid = []
    for s in range(env.nS):
        r, c = env.s2rc(s)
        if (r, c) == env.goal:
            grid.append('G')
        else:
            grid.append(arrows[np.argmax(Q[s])])
    for r in range(env.rows):
        print("  ".join(grid[r*env.cols:(r+1)*env.cols]))

    # Simple plot: episode returns
    plt.figure()
    plt.plot(returns, linewidth=1)
    plt.title("SARSA on MiniCliffWorld: Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
