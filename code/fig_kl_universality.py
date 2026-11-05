"""Figure: sharp boundary D(K, 1) for fixed NTPs vs per-segment evidence lambda = ell*KL/log n.

For a fixed NTP P and blocks of length ell = c log n, the block cumulant is K = c log m_P
(Theorem 5.4, fixed-NTP instance, a = 1). Choosing c = lambda / KL(F_P || Unif) puts every
NTP at the same per-segment evidence lambda = K'(1). The small-freedom limit (Corollary 5.5)
is K = (lambda / phi'(1)) * phi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

THETA = np.linspace(0.0, 30.0, 1501)   # only theta >= 0 enters D


def log_m(P):
    """log E_U[f_P(Y)^theta] on THETA, f_P(y) = sum_w y^(1/P_w - 1); substitute y = e^{-x}."""
    P = np.asarray(P, float)
    out = []
    for th in THETA:
        g = lambda x: np.exp(-x) * np.sum(np.exp(-x * (1 / P - 1))) ** th
        out.append(np.log(integrate.quad(g, 0, np.inf, limit=500)[0]))
    return np.array(out)


def kl(P):
    P = np.asarray(P, float)
    f = lambda y: np.sum(y ** (1 / P - 1))
    return integrate.quad(lambda y: f(y) * np.log(f(y)), 0, 1, limit=500)[0]


def D_of_K(K, a=1.0):
    t0 = (K[1] - K[0]) / (THETA[1] - THETA[0])
    ts = np.linspace(t0, t0 + 20, 3000)
    I = np.max(np.outer(ts, THETA) - K[None, :], axis=1)
    return a / 2 + np.max(ts - I + np.minimum(I, a) / 2)


PHI = np.array([integrate.quad(lambda w: ((1 + w) ** t - 1) / w, 0, 1)[0] - t for t in THETA])
PHI1 = np.pi ** 2 / 12 + 2 * np.log(2) - 2

NTPS = {
    "two-point, $\\Delta=0.1$": [0.9, 0.1],
    "two-point, $\\Delta=0.5$": [0.5, 0.5],
    "top 0.6 + 4$\\times$0.1": [0.6, 0.1, 0.1, 0.1, 0.1],
    "uniform over 20": [0.05] * 20,
    "top 0.9 + 10$\\times$0.01": [0.9] + [0.01] * 10,
}

if __name__ == "__main__":
    lams = np.linspace(0.0, 1.15, 47)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(lams, [0.5] + [D_of_K(l / PHI1 * PHI) for l in lams[1:]], "k-", lw=2.2,
            label="small-freedom limit ($K=\\kappa\\varphi$)")
    for name, P in NTPS.items():
        lm, k = log_m(P), kl(P)
        ax.plot(lams, [0.5] + [D_of_K(l / k * lm) for l in lams[1:]], lw=1.4, label=name)
    ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="order-blind boundary $a/2$")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("per-segment evidence $\\lambda=\\ell\\cdot\\mathrm{KL}/\\log n$")
    ax.set_ylabel("largest detectable $p$,  $D(K,1)$")
    ax.set_xlim(0, 1.15); ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig("../figures/boundary_kl.png", dpi=200)
    print("saved figures/boundary_kl.png")
