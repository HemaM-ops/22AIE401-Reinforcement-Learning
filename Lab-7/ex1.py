
import numpy as np
import matplotlib.pyplot as plt

# 4x4 deterministic GridWorld for TD(0) prediction (random policy)
class GridWorld4x4:
    def __init__(self, rows=4, cols=4, start=(0,0), goal=(3,3), step_reward=-1.0, gamma=0.9):
        self.rows, self.cols = rows, cols
        self.start, self.goal = start, goal
        self.step_reward, self.gamma = step_reward, gamma
        self.nS, self.nA = rows*cols, 4  # 0=Up,1=Right,2=Down,3=Left

    def s2rc(self, s): return divmod(s, self.cols)
    def rc2s(self, r, c): return r*self.cols + c
    def reset(self): return self.rc2s(*self.start)
    def is_terminal(self, s): return s == self.rc2s(*self.goal)

    def step(self, s, a):
        if self.is_terminal(s): return s, 0.0, True
        r, c = self.s2rc(s)
        if a == 0: r = max(0, r-1)
        elif a == 1: c = min(self.cols-1, c+1)
        elif a == 2: r = min(self.rows-1, r+1)
        elif a == 3: c = max(0, c-1)
        s_next = self.rc2s(r, c)
        reward = 0.0 if self.is_terminal(s_next) else self.step_reward
        done = self.is_terminal(s_next)
        return s_next, reward, done

def td0_prediction(env, alpha=0.1, episodes=100):
    V = np.zeros(env.nS)
    deltas = []
    for ep in range(episodes):
        s, done = env.reset(), False
        V_prev = V.copy()
        while not done:
            a = np.random.randint(env.nA)        # random policy
            s_next, r, done = env.step(s, a)
            V[s] += alpha * (r + env.gamma * V[s_next] - V[s])
            s = s_next
        deltas.append(np.max(np.abs(V - V_prev)))
    return V, deltas

if __name__ == "__main__":
    env = GridWorld4x4(gamma=0.9)
    V, deltas = td0_prediction(env, alpha=0.1, episodes=200)

    print("Final V(s) estimated by TD(0):")
    print(V.reshape(4,4))

    # Simple plot: convergence indicator
    plt.figure()
    plt.plot(deltas, linewidth=2)
    plt.title("TD(0) Prediction: Max |ΔV| per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Max |ΔV|")
    plt.grid(True)
    plt.show()
