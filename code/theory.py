"""Theoretical boundaries of the paper (Section 5).

D(K, a) = a/2 + sup_{t >= t0} [ t - I(t) + min(I(t), a)/2 ],  I = Legendre transform of K.
K_full = kappa * phi  (oracle block LLR), K_u (key-only block counts, Lemma 3.3).
"""
import numpy as np
from functools import lru_cache
from scipy import integrate

THETA = np.linspace(0.0, 60.0, 6001)       # upper-tail Legendre transforms only need theta >= 0


@lru_cache(maxsize=None)
def _phi_grid():
    f = lambda th: integrate.quad(lambda w: ((1 + w) ** th - 1) / w, 0, 1)[0] - th
    return np.array([f(t) for t in THETA])


def phi(theta):
    return np.interp(theta, THETA, _phi_grid())


def K_count(kappa, u):
    rho = u + 1 - np.exp(-u)
    return kappa * (u * ((rho / u) ** THETA - 1) - THETA * (rho - u))


def D_of_K(Kvals, a, n_t=3000):
    """Sharp boundary D(K, a) from K evaluated on THETA (K(0) = K(1) = 0)."""
    t0 = (Kvals[1] - Kvals[0]) / (THETA[1] - THETA[0])
    ts = np.linspace(t0, t0 + 40, n_t)
    I = np.max(np.outer(ts, THETA) - Kvals[None, :], axis=1)
    return a / 2 + np.max(ts - I + np.minimum(I, a) / 2)


def D_full(kappa, a):
    return D_of_K(kappa * _phi_grid(), a)


def D_cnt(kappa, a, us=None):
    us = np.exp(np.linspace(np.log(0.05), np.log(5), 30)) if us is None else us
    return max(D_of_K(K_count(kappa, u), a) for u in us)


if __name__ == "__main__":
    for a in [0.5, 0.7]:
        print(f"a={a}: kappa0={a/(2*0.8450557556638942-0.5):.3f}, kappa*={a/0.20876139454400366:.3f}")
        for k in [0.5, 1, 2, 3]:
            print(f"   kappa={k}: D_full={D_full(k, a):.4f}  D_cnt={D_cnt(k, a):.4f}  a/2={a/2}")
