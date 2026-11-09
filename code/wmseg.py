"""Simulation tools for Gumbel-max watermark detection under segment survival.

Model (paper, Assumption 2.2): pivots Y_t are independent; Y_t ~ Unif(0,1) off the
watermarked set S, and Y_t ~ F_{P_t} on S, where Y = V ** P_W with W ~ P_t, V ~ Unif(0,1)
(Lemma 2.1). The least-favorable NTP is the two-point law P = (1 - Delta, Delta).
"""
import numpy as np
from scipy import stats


# ------------------------------------------------------------------ pivots
def sample_pivots(n, S_mask, delta, rng):
    """Pivots for one text; two-point NTP with constant Delta on the watermarked set."""
    y = rng.random(n)
    k = int(S_mask.sum())
    if k:
        v = rng.random(k)
        expo = np.where(rng.random(k) < delta, delta, 1.0 - delta)
        y[S_mask] = v ** expo
    return y


def sample_pivots_hetero(n, S_mask, deltas, n_rest, rng):
    """Heterogeneous NTPs: token t has top probability 1 - deltas[t]; the remaining mass
    deltas[t] is split equally over n_rest[t] tokens. Returns pivots."""
    y = rng.random(n)
    idx = np.flatnonzero(S_mask)
    if idx.size:
        d = deltas[idx]
        v = rng.random(idx.size)
        top = rng.random(idx.size) >= d
        expo = np.where(top, 1.0 - d, d / n_rest[idx])
        y[idx] = v ** expo
    return y


def segment_mask(n, n_surv, seg_len, rng, aligned=True):
    """Watermarked set = n_surv // seg_len disjoint segments of length seg_len.
    seg_len = 1 gives the i.i.d.-survival model; aligned=True uses the grid of length seg_len."""
    mask = np.zeros(n, dtype=bool)
    m = max(n_surv // seg_len, 1)
    if aligned:
        for b in rng.choice(n // seg_len, size=m, replace=False):
            mask[b * seg_len:(b + 1) * seg_len] = True
    else:
        free = n - m * seg_len
        starts = np.sort(rng.choice(free + m, size=m, replace=False))
        for i, s in enumerate(starts):
            st = s - i + i * seg_len
            mask[st:st + seg_len] = True
    return mask


# ------------------------------------------------------------------ HC variants
def hc_plus(pvals, alpha0=0.5):
    """Donoho-Jin HC^+: max over k <= alpha0 N with p_(k) > 1/N."""
    N = pvals.size
    p = np.sort(pvals)
    k = np.arange(1, N + 1)
    keep = (k <= alpha0 * N) & (p > 1.0 / N) & (p < 1.0)
    if not keep.any():
        return 0.0
    z = np.sqrt(N) * (k / N - p) / np.sqrt(p * (1 - p))
    return float(np.max(z[keep]))


def hc_dyadic(pvals):
    """Dyadic HC of the paper's test psi*: max_j (#{p <= 2^-j} - N 2^-j) / sqrt(N 2^-j)."""
    N = pvals.size
    J = int(np.floor(np.log2(N)))
    lv = 2.0 ** -np.arange(1, J + 1)
    cnt = np.searchsorted(np.sort(pvals), lv, side="right")
    return float(np.max((cnt - N * lv) / np.sqrt(N * lv)))


def psi_star_stats(pvals):
    """Components of psi*: (dyadic HC, -log10 of the Bonferroni-scaled min p-value)."""
    return hc_dyadic(pvals), float(-np.log10(pvals.size * np.min(pvals)))


# ------------------------------------------------------------------ order-blind statistics
def stat_sum(y):
    """Aaronson's sum statistic sum -log(1 - Y_t)."""
    return float(np.sum(-np.log1p(-y)))


def stat_token_hc(y):
    """HC^+ on token p-values 1 - Y_t (the HC member of the Tr-GoF family)."""
    return hc_plus(1.0 - y)


# ------------------------------------------------------------------ block statistics
def blocks(y, L, offset=0):
    y = y[offset:]
    nb = y.size // L
    return y[:nb * L].reshape(nb, L)


def count_pvalues(y, L, thresh, rng, offset=0):
    """Randomized exact binomial p-values of block counts #{Y > 1 - thresh} (key-only)."""
    c = (blocks(y, L, offset) > 1.0 - thresh).sum(axis=1)
    sf = stats.binom.sf(c - 1, L, thresh)
    pmf = stats.binom.pmf(c, L, thresh)
    return np.clip(sf - rng.random(c.size) * pmf, 1e-300, 1.0)


def log_f_two_point(y, delta):
    """log f_Delta(y) = log(y^alpha + y^beta), alpha = Delta/(1-Delta), beta = 1/Delta - 1."""
    ly = np.log(y)
    return np.logaddexp(delta / (1 - delta) * ly, (1 / delta - 1) * ly)


def block_llr(y, L, delta, offset=0):
    """Oracle block log-likelihood ratios T_k (two-point NTP)."""
    return log_f_two_point(blocks(y, L, offset), delta).sum(axis=1)


class NullLLR:
    """Monte Carlo null distribution of the block LLR, for block p-values."""
    def __init__(self, L, delta, rng, M=200000, chunk=20000):
        vals = []
        for _ in range(M // chunk):
            vals.append(log_f_two_point(rng.random((chunk, L)), delta).sum(axis=1))
        self.sorted = np.sort(np.concatenate(vals))
        self.M = self.sorted.size

    def pvalues(self, T):
        # P(T_null >= T), with +1 smoothing so p-values are valid
        ge = self.M - np.searchsorted(self.sorted, T, side="left")
        return (ge + 1.0) / (self.M + 1.0)
