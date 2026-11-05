"""E2: finite-n phase diagram in (p, kappa) at the critical scale, vs. theory.

n = 2^18 tokens, blocks of length ell = 512, so N = 512 = n^{1/2} (a = 0.5).
Delta = kappa * log(n) / ell, eps = n^{-p}, m = round(eps * N) aligned signal segments.
Theory: order-blind boundary p = a/2; oracle boundary D_full(kappa, a); key-only D_cnt.
"""
import sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wmseg as W, harness as H
sys.path.insert(0, ".")
from theory import D_full, D_cnt

out = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
rng = np.random.default_rng(2)
n, ell, alpha = 2 ** 18, 512, 0.05
N, Lg = n // ell, np.log(n)
a = np.log(N) / np.log(n)
kappas = [0.25, 0.5, 1.0, 2.0, 3.0]
ps = [0.25, 0.30, 0.35, 0.40, 0.45]

res = {}; t0 = time.time()
for kap in kappas:
    delta = kap * Lg / ell
    thresh = 1.3 * delta
    nullT = W.NullLLR(ell, delta, rng, M=200000)
    def test(y):
        return {"token HC (order-blind)": (W.stat_token_hc(y),),
                "count psi* (key-only)": W.psi_star_stats(W.count_pvalues(y, ell, thresh, rng)),
                "oracle LLR psi*": W.psi_star_stats(nullT.pvalues(W.block_llr(y, ell, delta)))}
    null = [test(W.sample_pivots(n, np.zeros(n, bool), delta, rng)) for _ in range(reps)]
    crit = H.calibrate(null, alpha)
    for p in ps:
        m = max(1, int(round(n ** (-p) * N)))
        alt = []
        for _ in range(reps):
            S = np.zeros(n, bool)
            for b in rng.choice(N, size=m, replace=False):
                S[b * ell:(b + 1) * ell] = True
            alt.append(test(W.sample_pivots(n, S, delta, rng)))
        res[f"{kap},{p}"] = dict(H.power(alt, crit), m=m, delta=delta)
        print(f"kappa={kap} p={p} m={m} delta={delta:.4f}:",
              {k: round(v, 2) for k, v in res[f'{kap},{p}'].items() if k not in ('m', 'delta')},
              f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)

names = [k for k in next(iter(res.values())) if k not in ("m", "delta")]
kk = np.linspace(0.3, 3.2, 40)
Dfull = [D_full(k, a) for k in kk]; Dcnt = [D_cnt(k, a) for k in kk]
fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), sharey=True)
for ax, nm in zip(axes, names):
    Z = np.array([[res[f"{k},{p}"][nm] for k in kappas] for p in ps])
    im = ax.imshow(Z, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                   extent=[0, len(kappas), ps[0] - 0.025, ps[-1] + 0.025])
    for i, p in enumerate(ps):
        for j in range(len(kappas)):
            ax.text(j + 0.5, p, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if Z[i, j] < 0.6 else "k")
    xmap = lambda k: np.interp(k, kappas, np.arange(len(kappas)) + 0.5)
    ax.plot(xmap(kk), Dfull, "r-", lw=1.5, label=r"theory $D_{\rm full}$")
    ax.plot(xmap(kk), Dcnt, "r--", lw=1.2, label=r"theory $D_{\rm cnt}$")
    ax.axhline(a / 2, color="w", ls=":", lw=1.2, label="order-blind $a/2$")
    ax.set_xticks(np.arange(len(kappas)) + 0.5); ax.set_xticklabels(kappas)
    ax.set_xlabel(r"$\kappa$"); ax.set_title(nm, fontsize=10); ax.set_ylim(ps[0] - 0.025, ps[-1] + 0.025)
axes[0].set_ylabel(r"sparsity $p$ ($\epsilon=n^{-p}$)"); axes[0].legend(fontsize=7, loc="upper left")
fig.colorbar(im, ax=axes, shrink=0.8, label="power at level 0.05")
fig.suptitle(rf"E2: $n=2^{{18}}$, $\ell=N=512$ ($a=0.5$), {reps} reps; detectable = below the curve", fontsize=9)
fig.savefig(out, dpi=180, bbox_inches="tight"); print("saved", out)
