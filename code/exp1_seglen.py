"""E1: power vs. segment length at a fixed number of surviving watermarked tokens.

Prediction: order-blind tests (sum, token HC) are flat in the segment length (Prop. 4.1);
segment-aware tests (key-only count psi*, oracle block-LLR psi*) gain power as the expected
number of events per segment, seg_len * Delta, grows (Thm. 4.2, Thm. 5.2).
"""
import sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wmseg as W, harness as H

out = sys.argv[1]
rng = np.random.default_rng(1)
n, delta, alpha, reps = 50000, 0.1, 0.05, 400
seg_lens, n_survs = [1, 25, 100, 400], [100, 200, 400, 800]
thresh = 1.3 * delta

def make_tests(L, nullT):
    def f(y):
        sm = W.stat_sum(y)
        cnt = W.psi_star_stats(W.count_pvalues(y, L, thresh, rng))
        return {"sum": (sm,),
                "token HC": (W.stat_token_hc(y),),
                "count psi*": cnt,
                "combined (sum OR count psi*)": (sm,) + cnt,
                "oracle LLR psi*": W.psi_star_stats(nullT.pvalues(W.block_llr(y, L, delta)))}
    return f

t0 = time.time(); res = {}
for L in seg_lens:
    nullT = W.NullLLR(L, delta, rng, M=200000)
    test = make_tests(L, nullT)
    null = [test(W.sample_pivots(n, np.zeros(n, bool), delta, rng)) for _ in range(reps)]
    crit = H.calibrate(null, alpha)
    for k in n_survs:
        if k < L: continue
        alt = [test(W.sample_pivots(n, W.segment_mask(n, k, L, rng), delta, rng)) for _ in range(reps)]
        res[f"{L},{k}"] = H.power(alt, crit)
        print(L, k, {a: round(b, 2) for a, b in res[f"{L},{k}"].items()}, flush=True)
print(f"{time.time()-t0:.0f}s")
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)

names = list(next(iter(res.values())).keys())
fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), sharey=True)
for ax, nm in zip(axes, names):
    for L in seg_lens:
        xs = [k for k in n_survs if f"{L},{k}" in res]
        ax.plot(xs, [res[f"{L},{k}"][nm] for k in xs], "o-", label=rf"$\ell={L}$ ({L*delta:g} ev/seg)")
    ax.set_xscale("log"); ax.set_xticks(n_survs); ax.set_xticklabels(n_survs)
    ax.set_title(nm, fontsize=10); ax.set_xlabel("surviving tokens"); ax.axhline(alpha, color="gray", ls=":")
axes[0].set_ylabel("power (level 0.05)"); axes[-1].legend(fontsize=7)
fig.suptitle(rf"E1: $n={n}$, two-point NTP $\Delta={delta}$, aligned segments; order-blind (left two), segment-aware, combined, oracle", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=180); print("saved", out)
