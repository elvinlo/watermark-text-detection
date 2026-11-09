"""Schematic of the coarse phase diagram (Theorem 5.2) in the (q, p) plane for fixed gamma."""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
gam = 0.4
q = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for ax, seg in zip(axes, [False, True]):
    ax.fill_between(q, 0, (1 - q) / 2, color="tab:blue", alpha=0.35, label="detectable by order-blind tests ($2p+q<1$)")
    if seg:
        qq = q[q < gam]
        ax.fill_between(qq, (1 - qq) / 2, 1 - gam, color="tab:orange", alpha=0.45,
                        label=r"extra region from contiguity ($\gamma>q$)")
        ax.axhspan(1 - gam, 1, color="lightgray", alpha=0.6, label=r"fewer than one segment ($p+\gamma>1$)")
        ax.axvline(gam, color="k", ls=":", lw=1)
        ax.text(gam + 0.01, 0.62, r"$q=\gamma$", fontsize=8)
        ax.set_title(rf"segment survival, segment length $\ell=n^{{\gamma}}$, $\gamma={gam}$", fontsize=9)
    else:
        ax.set_title("i.i.d. survival (Li et al.)", fontsize=9)
    ax.plot(q, (1 - q) / 2, "k-", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel(r"$q$  (event rate $\Delta=n^{-q}$; rarer events to the right)")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel(r"$p$  (surviving fraction $\epsilon=n^{-p}$)")
fig.tight_layout(); fig.savefig(out, dpi=200); print("saved", out)
