# Code for "When Does Contiguity Help Watermark Detection?"

All experiments are synthetic. They simulate Gumbel-max pivots exactly under the paper's pivot model (Assumption A1).

## Requirements
Python 3.9+, with `numpy`, `scipy`, and `matplotlib`. Each script fixes its random seed.

## Library modules
| File | Contents |
|---|---|
| `wmseg.py` | Pivot simulation (two-point NTPs, and heterogeneous NTPs for E4; aligned or random segments, or i.i.d. survival). Tests: sum, token-level HC, block counts, oracle block log-likelihood ratios, the block HC test ψ* (dyadic HC + Bonferroni), and the combined test. |
| `harness.py` | Monte Carlo calibration of critical values under H0 and power estimation. Multi-component tests are calibrated as Bonferroni unions. |
| `theory.py` | Numerical sharp boundaries: D(K, a) via Legendre transforms, D_full (K = κφ), and D_cnt (key-only counts). Run `python theory.py` to print the Table 2 values. |

## Figures and tables
Run each script from inside `code/`. Each writes the figure, and for experiments a `.json` file with the raw power estimates next to it.

| Paper item | Command |
|---|---|
| Figure 1 (coarse phase diagram) | `python fig_coarse.py ../figures/coarse_phase.png` |
| Figure 2 (boundary vs per-segment KL) | `python fig_kl_universality.py` (writes `../figures/boundary_kl.png`) |
| Table 2 (small-freedom boundaries) | `python theory.py` |
| Figure 3, E1 (segment length) | `python exp1_seglen.py ../figures/e1_seglen.png` |
| Figure 4, E2 (finite-n phase diagram) | `python exp2_phase.py ../figures/e2_phase.png 300` (the second argument is the number of replications; about 5 minutes on a laptop) |
| Appendix figure, E3 (unknown alignment) | `python exp3_unaligned.py ../figures/e3_unaligned.png` |
| Appendix figure, E4 (heterogeneous NTPs) | `python exp4_hetero.py ../figures/e4_hetero.png` |
| Appendix figure, E5 (essay-length n = 2000) | `python exp5_short.py ../figures/e5_short.png` (2000 replications by default; about 30 seconds) |

`exp_power_vs_seglen.py` is an earlier version of E1, kept for reference. The paper does not use it.

## Calibration details
- Level 0.05, with 300–400 Monte Carlo null replications per setting; the standard error of the power estimates is at most 0.03.
- Key-only block counts use the threshold 1 − 1.3Δ, with exact randomized binomial p-values.
- Oracle block log-likelihood ratios use Monte Carlo null p-values.
- The combined test is a calibrated Bonferroni union of its three components: the sum test and the two components of the count-based ψ* (dyadic HC and Bonferroni). Each component is tested at level α/3 using its Monte Carlo null quantile.

