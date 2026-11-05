"""E4: heterogeneous next-token distributions (beyond the two-point model).

Each token is near-deterministic (Delta_t = 0.01) w.p. 0.6, otherwise Delta_t ~ Unif(0.1, 0.5);
the non-top mass is split equally over n_rest in {1, 2, 5, 20} tokens. Mean Delta_t ~ 0.13.
The detector does not know Delta_t: the key-only count test scans thresholds
s in {0.025, 0.05, 0.1, 0.2} (max over thresholds, calibrated under the null).
Prediction: order-blind tests stay flat in the segment length; the count test gains with the
expected number of events per segment, sum of Delta_t over a segment.
"""
import sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wmseg as W, harness as H

out = sys.argv[1]
rng = np.random.default_rng(4)
n, alpha, reps, n_surv = 50000, 0.05, 400, 800
seg_lens = [1, 25, 100, 400]
threshs = [0.025, 0.05, 0.1, 0.2]

def draw_ntp():
    near = rng.random(n) < 0.6
    deltas = np.where(near, 0.01, rng.uniform(0.1, 0.5, n))
    n_rest = rng.choice([1, 2, 5, 20], size=n)
    return deltas, n_rest

def count_multi(y, L):
    st = [W.psi_star_stats(W.count_pvalues(y, L, s, rng)) for s in threshs]
    return tuple(np.max(st, axis=0))

res = {}; t0 = time.time()
for L in seg_lens:
    def test(y):
        sm, cm = W.stat_sum(y), count_multi(y, L)
        return {"sum": (sm,), "token HC": (W.stat_token_hc(y),),
                "count psi* (unknown Delta)": cm,
                "combined (sum OR count psi*)": (sm,) + cm}
    null = [test(rng.random(n)) for _ in range(reps)]
    crit = H.calibrate(null, alpha)
    alt = []
    for _ in range(reps):
        d, k = draw_ntp()
        alt.append(test(W.sample_pivots_hetero(n, W.segment_mask(n, n_surv, L, rng), d, k, rng)))
    res[L] = H.power(alt, crit)
    print(L, {a: round(b, 2) for a, b in res[L].items()}, f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)

plt.figure(figsize=(5.2, 3.5))
for nm in res[seg_lens[0]]:
    plt.plot(seg_lens, [res[L][nm] for L in seg_lens], "o-", label=nm)
plt.xscale("log"); plt.xticks(seg_lens, seg_lens); plt.axhline(alpha, color="gray", ls=":")
plt.xlabel("segment length (expected events/segment = 0.13 x length)"); plt.ylabel("power (level 0.05)")
plt.title(rf"E4: heterogeneous NTPs, $n={n}$, {n_surv} surviving tokens", fontsize=9)
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out, dpi=180); print("saved", out)
