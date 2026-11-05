"""E3: unknown alignment and unknown segment length (Prop. 5.4).

Segments of true length 100 are placed at random (unaligned). Compare the key-only count psi*
(i) with an aligned oracle (aligned data, grid = true length), (ii) a single grid of the true
length on unaligned data, (iii) 4 shifted grids of length 80, and (iv) a fully adaptive
multiscale scan over lengths {25, 50, 100, 200} x 4 shifts. Multiplicity is handled by
calibrating the max over grids under the null.
"""
import sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wmseg as W, harness as H

out = sys.argv[1]
rng = np.random.default_rng(3)
n, delta, alpha, reps, seg = 50000, 0.1, 0.05, 400, 100
thresh = 1.3 * delta
n_survs = [200, 400, 800, 1600]

def grid_stats(y, L, shifts):
    hcs, bonf = [], []
    for s in shifts:
        h, b = W.psi_star_stats(W.count_pvalues(y, L, thresh, rng, offset=s))
        hcs.append(h); bonf.append(b)
    return max(hcs), max(bonf)

def test(y):
    return {"token HC (order-blind)": (W.stat_token_hc(y),),
            "single grid, true length": grid_stats(y, seg, [0]),
            "4 shifted grids, length 80": grid_stats(y, 80, [0, 20, 40, 60]),
            "multiscale (length unknown)": tuple(np.max([grid_stats(y, L, [0, L // 4, L // 2, 3 * L // 4])
                                                         for L in (25, 50, 100, 200)], axis=0))}

t0 = time.time()
null = [test(W.sample_pivots(n, np.zeros(n, bool), delta, rng)) for _ in range(reps)]
crit = H.calibrate(null, alpha)
# aligned oracle reference (aligned data, grid = true length)
null_al = [{"aligned oracle": grid_stats(W.sample_pivots(n, np.zeros(n, bool), delta, rng), seg, [0])} for _ in range(reps)]
crit_al = H.calibrate(null_al, alpha)
res = {}
for k in n_survs:
    alt = [test(W.sample_pivots(n, W.segment_mask(n, k, seg, rng, aligned=False), delta, rng)) for _ in range(reps)]
    alt_al = [{"aligned oracle": grid_stats(W.sample_pivots(n, W.segment_mask(n, k, seg, rng, aligned=True), delta, rng), seg, [0])} for _ in range(reps)]
    res[k] = dict(H.power(alt_al, crit_al), **H.power(alt, crit))
    print(k, {a: round(b, 2) for a, b in res[k].items()}, f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)

plt.figure(figsize=(5.5, 3.6))
for nm in res[n_survs[0]]:
    plt.plot(n_survs, [res[k][nm] for k in n_survs], "o-", label=nm)
plt.xscale("log"); plt.xticks(n_survs, n_survs); plt.axhline(alpha, color="gray", ls=":")
plt.xlabel("surviving tokens (segments of length 100)"); plt.ylabel("power (level 0.05)")
plt.title(rf"E3: unaligned segments, $n={n}$, $\Delta={delta}$ (10 events/segment)", fontsize=9)
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out, dpi=180); print("saved", out)
