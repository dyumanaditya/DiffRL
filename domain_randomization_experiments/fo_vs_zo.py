import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Toy environment& helpers
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_reward(u, p, eps):
    """Smoothed Heaviside: ≈1[u>p] with derivative."""
    return sigmoid((u - p) / eps)

# -------------------------
# Simulation
# -------------------------
def run_sim(steps=800, eps=0.05, sigma=0.2, lr_fo=0.002, lr_zo=0.05, seed=0):
    rng = np.random.default_rng(seed)
    θ_fo, θ_zo = -1.0, -1.0

    R_fo, R_zo = [], []
    g2_fo, g2_zo = [], []

    for _ in range(steps):
        # Domain randomisation: new hidden parameter each step
        p = rng.uniform(-1.0, 1.0)

        # ----- First‑Order update -----
        r_fo      = step_reward(θ_fo, p, eps)
        grad_fo   = (1/eps) * r_fo * (1 - r_fo)      # analytic ∂r/∂θ
        θ_fo     += lr_fo * grad_fo
        g2_fo.append(grad_fo**2)

        # ----- Zero‑Order update -----
        u         = rng.normal(θ_zo, sigma)
        r         = step_reward(u, p, eps)
        grad_zo   = (u - θ_zo) / sigma**2 * r        # score‑function estimator
        θ_zo     += lr_zo * grad_zo
        g2_zo.append(grad_zo**2)

        # Evaluate expected reward over fresh envs
        p_eval    = rng.uniform(-1.0, 1.0, 2000)
        R_fo.append(np.mean(step_reward(θ_fo, p_eval, eps)))
        R_zo.append(np.mean(step_reward(θ_zo, p_eval, eps)))

    return dict(R_fo=np.array(R_fo),
                R_zo=np.array(R_zo),
                g2_fo=np.array(g2_fo),
                g2_zo=np.array(g2_zo),
                θ_fo=θ_fo, θ_zo=θ_zo)

stats = run_sim()

# -------------------------
# Plot learning curves
# -------------------------
plt.figure()
plt.plot(stats["R_fo"], label="First‑Order")
plt.plot(stats["R_zo"], label="Zero‑Order")
plt.xlabel("Training step")
plt.ylabel("Expected reward")
plt.ylim(0, 1.05)
plt.title("Domain randomization\nFO converges **slower** than ZO on a discontinuous task")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Print headline statistics
# -------------------------
print("Final θ(FO):", round(stats["θ_fo"], 3))
print("Final θ(ZO):", round(stats["θ_zo"], 3))
print("Final expected reward (FO):", round(stats["R_fo"][-1], 4))
print("Final expected reward (ZO):", round(stats["R_zo"][-1], 4))
print("\nAverage gradient variance during training")
print("  FO:", round(stats["g2_fo"].mean(), 4))
print("  ZO:", round(stats["g2_zo"].mean(), 4))
