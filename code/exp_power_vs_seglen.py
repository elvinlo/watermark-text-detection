"""Experiment 1: power vs. segment length at a fixed number of surviving watermarked tokens.

Prediction (Prop. 4.1, Thm. 4.3, Thm. 5.3): order-blind tests (sum, token HC) have the same
power for every segment length, while order-aware block tests gain power as the expected
number of watermark events per segment, seg_len * Delta, grows.
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wmseg as W

out = sys.argv[1] if len(sys.argv) > 1 else "power_vs_seglen.png"
rng = np.random.default_rng(236)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
alpha, reps = 0.05, 400
thresh = 1.3 * delta                     # count pivots above 1 - 1.3 * Delta (u* ~ 1.3)
seg_lens = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 20, 80, 320]
n_survs = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [100, 200, 400, 800]

def all_stats(y, L):
    return {"sum (order-blind)": W.stat_sum(y),
            "token HC (order-blind)": W.stat_token_hc(y),
            "block-count HC": W.stat_block_count_hc(y, L, thresh, rng),
            "block scan": W.stat_block_scan(y, L, thresh)}

t0 = time.time()
null_mask = np.zeros(n, dtype=bool)
crit = {}
for L in seg_lens:
    vals = [all_stats(W.sample_pivots(n, null_mask, delta, rng), L) for _ in range(reps)]
    crit[L] = {k: np.quantile([v[k] for v in vals], 1 - alpha) for k in vals[0]}

power = {}
for L in seg_lens:
    for k_surv in n_survs:
        if k_surv < L:
            continue
        rej = {}
        for _ in range(reps):
            S = W.segment_mask(n, k_surv, L, rng, aligned=True)
            st = all_stats(W.sample_pivots(n, S, delta, rng), L)
            for name, v in st.items():
                rej[name] = rej.get(name, 0) + (v > crit[L][name])
        power[(L, k_surv)] = {name: r / reps for name, r in rej.items()}
        print(f"seg_len={L:4d} (events/seg={L*delta:5.2f}) surv={k_surv:4d}: " +
              ", ".join(f"{nm.split(' (')[0]}={p:.2f}" for nm, p in power[(L, k_surv)].items()),
              flush=True)
print(f"done in {time.time()-t0:.0f}s")

names = list(next(iter(power.values())).keys())
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), sharey=True)
for ax, nm in zip(axes, names):
    for L in seg_lens:
        xs = [k for k in n_survs if (L, k) in power]
        ax.plot(xs, [power[(L, k)][nm] for k in xs], "o-", label=f"seg len {L} ({L*delta:g} ev/seg)")
    ax.set_xscale("log"); ax.set_title(nm, fontsize=10); ax.set_xlabel("surviving watermarked tokens")
    ax.axhline(alpha, color="gray", ls=":")
axes[0].set_ylabel(f"power (level {alpha})"); axes[-1].legend(fontsize=7)
fig.suptitle(rf"$n={n}$, two-point NTP with $\Delta={delta}$; aligned segments", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=180)
print("saved", out)
