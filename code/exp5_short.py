"""E5: short-document check (essay-length text, n = 2000 tokens).

Same design as E1, but at an essay-like length and for two evidence levels per token:
Delta = 0.1 (per-token KL ~ 0.03 nats, low-temperature / code-like) and
Delta = 0.4 (per-token KL ~ 0.17 nats, closer to open-ended prose).
Here N = n / ell is small (80 sentence blocks at ell = 25), so the saturation level
log N ~ 4.4 nats is reached by one sentence at Delta = 0.4 but needs ~140 tokens at Delta = 0.1.
"""
import sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wmseg as W, harness as H

out = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(5)
n, alpha = 2000, 0.05
deltas, seg_lens, n_survs = [0.1, 0.4], [1, 25, 100], [50, 100, 200, 400]

def make_tests(L, delta, nullT):
    thresh = 1.3 * delta
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
for delta in deltas:
    for L in seg_lens:
        nullT = W.NullLLR(L, delta, rng, M=200000)
        test = make_tests(L, delta, nullT)
        null = [test(W.sample_pivots(n, np.zeros(n, bool), delta, rng)) for _ in range(reps)]
        crit = H.calibrate(null, alpha)
        for k in n_survs:
            if k < L: continue
            alt = [test(W.sample_pivots(n, W.segment_mask(n, k, L, rng), delta, rng)) for _ in range(reps)]
            res[f"{delta},{L},{k}"] = H.power(alt, crit)
            print(delta, L, k, {a: round(b, 2) for a, b in res[f"{delta},{L},{k}"].items()}, flush=True)
print(f"{time.time()-t0:.0f}s")
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)

names = list(next(iter(res.values())).keys())
fig, axes = plt.subplots(len(deltas), len(names), figsize=(3.0 * len(names), 2.8 * len(deltas)), sharey=True)
for r, delta in enumerate(deltas):
    for ax, nm in zip(axes[r], names):
        for L in seg_lens:
            xs = [k for k in n_survs if f"{delta},{L},{k}" in res]
            ax.plot(xs, [res[f"{delta},{L},{k}"][nm] for k in xs], "o-", label=rf"$\ell={L}$")
        ax.set_xscale("log"); ax.set_xticks(n_survs); ax.set_xticklabels(n_survs); ax.minorticks_off()
        ax.axhline(alpha, color="gray", ls=":")
        if r == 0: ax.set_title(nm, fontsize=9)
        if r == len(deltas) - 1: ax.set_xlabel("surviving tokens")
    axes[r][0].set_ylabel(rf"$\Delta={delta}$: power")
axes[0][-1].legend(fontsize=7)
fig.suptitle(rf"E5: short documents, $n={n}$, two-point NTP, aligned segments, {reps} reps", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=180); print("saved", out)
