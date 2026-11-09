"""Monte Carlo calibration/power harness.

Each test maps pivots y -> dict of component statistics (larger = more evidence).
A test with several components (e.g. psi* = dyadic HC OR Bonferroni) rejects if any
component exceeds its null (1 - alpha/m) quantile, i.e. a calibrated Bonferroni union.
"""
import numpy as np


def calibrate(null_stats, alpha):
    """null_stats: list of dicts {test: tuple(components)} -> critical values per test."""
    crit = {}
    for name in null_stats[0]:
        comps = np.array([s[name] for s in null_stats])          # reps x m
        m = comps.shape[1]
        crit[name] = np.quantile(comps, 1 - alpha / m, axis=0)
    return crit


def reject(stat, crit_vec):
    return bool(np.any(np.asarray(stat) > crit_vec))


def power(alt_stats, crit):
    return {name: float(np.mean([reject(s[name], crit[name]) for s in alt_stats])) for name in crit}
