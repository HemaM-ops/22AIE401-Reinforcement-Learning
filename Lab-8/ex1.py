
import numpy as np
import matplotlib.pyplot as plt

class StochasticGrid4x4:
    def __init__(self, slip=0.2, gamma=0.99):
        self.rows, self.cols = 4, 4
        self.nS, self.nA = self.rows*self.cols, 4  # 0=Up,1=Right,2=Down,3=Left
        self.start, self.goal = (0,0), (3,3)
        self.slip, self.gamma = slip, gamma

    def s2rc(self, s): return divmod(s, self.cols)
    def rc2s(self, r, c): return r*self.cols + c
    def reset(self): return self.rc2s(*self.start)
    def is_goal(self, s): return s == self.rc2s(*self.goal)

    def _move(self, r, c, a):
        if a == 0: r = max(0, r-1)
        elif a == 1: c = min(self.cols-1, c+1)
        elif a == 2: r = min(self.rows-1, r+1)
        elif a == 3: c = max(0, c-1)
        return r, c

    def step(self, s, a):
        r, c = self.s2rc(s)
        if np.random.rand() < 1 - self.slip:
            a_exec = a
        else:
            a_exec = np.random.choice([x for x in range(4) if x != a])
        r2, c2 = self._move(r, c, a_exec)
        s_next = self.rc2s(r2, c2)
        done = self.is_goal(s_next)
        reward = 10.0 if done else -1.0
        return s_next, reward, done

def epsilon_greedy(Q, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.random.choice(np.flatnonzero(np.isclose(Q[s], np.max(Q[s]))))

def q_learning(env, alpha=0.5, gamma=0.99, epsilon=0.1, episodes=2000, max_steps=200):
    Q = np.zeros((env.nS, env.nA))
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done, G = False, 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon)
            s_next, r, done = env.step(s, a)
            G += r
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next
            if done: break
        returns.append(G)
    return Q, returns

if __name__ == "__main__":
    env = StochasticGrid4x4()
    Q, returns = q_learning(env)

    arrows = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    policy = [arrows[np.argmax(Q[s])] if not env.is_goal(s) else 'G' for s in range(env.nS)]
    print("Greedy policy (rows top→bottom):")
    for r in range(env.rows):
        print("  ".join(policy[r*env.cols:(r+1)*env.cols]))

    plt.plot(returns)
    plt.title("Q-Learning: Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
