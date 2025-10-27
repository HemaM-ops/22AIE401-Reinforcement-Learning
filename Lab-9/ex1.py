
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Exercise 1: TD(λ) Prediction on 1D Random Walk (A..G)
# States 0 and 6 are terminal with rewards 0 and 1 respectively.
# Start at state 3 (D). Policy: random left/right with equal prob.
# γ = 1.0, initial V=0.5 for non-terminals, V(terminal)=true reward.
# ------------------------------

class RandomWalk7:
    def __init__(self):
        # States: 0:A (terminal 0), 1:B, 2:C, 3:D, 4:E, 5:F, 6:G (terminal 1)
        self.nS = 7
        self.start = 3
        self.terminals = {0:0.0, 6:1.0}
        self.gamma = 1.0

    def reset(self):
        return self.start

    def step(self, s):
        if s in self.terminals:
            return s, 0.0, True
        a = np.random.choice([-1, +1])  # left or right
        s_next = s + a
        done = s_next in self.terminals
        reward = self.terminals.get(s_next, 0.0)
        return s_next, reward, done

# True state values for non-terminal states under the random policy (linear)
TRUE_V = np.array([0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])

def td_lambda_prediction(env, lam=0.8, alpha=0.1, episodes=100):
    V = np.ones(env.nS) * 0.5
    V[0], V[6] = 0.0, 1.0  # terminal values fixed
    gamma = env.gamma
    rms_per_ep = []

    for ep in range(episodes):
        e = np.zeros(env.nS)   # eligibility traces (state-based)
        s = env.reset()
        done = False

        while not done:
            s_next, r, done = env.step(s)
            if done:
                delta = r - V[s]
            else:
                delta = r + gamma * V[s_next] - V[s]
            e[s] += 1.0
            V += alpha * delta * e
            e *= gamma * lam
            s = s_next

        # compute RMS error on non-terminals (1..5)
        rms = np.sqrt(np.mean((V[1:6] - TRUE_V[1:6])**2))
        rms_per_ep.append(rms)

    return V, np.array(rms_per_ep)

if __name__ == "__main__":
    env = RandomWalk7()
    lambdas = [0.0, 0.3, 0.6, 0.9, 1.0]
    episodes = 100
    alpha = 0.1

    plt.figure()
    for lam in lambdas:
        _, rms_curve = td_lambda_prediction(env, lam=lam, alpha=alpha, episodes=episodes)
        plt.plot(rms_curve, label=f"λ={lam}")
    plt.xlabel("Episode")
    plt.ylabel("RMS Error (non-terminals)")
    plt.title("TD(λ) Prediction on Random Walk (A..G)")
    plt.grid(True)
    plt.legend()
    plt.show()
