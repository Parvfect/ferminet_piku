---
name: tool-energy-convergence
description: tools/energy_convergence.py — assess VMC energy convergence from train_stats.csv via block-averaging + trailing-window drift/slope test
metadata: 
  node_type: memory
  type: reference
  originSessionId: 40812d63-6ed1-449d-bf5d-6c792162cfd4
---

## tools/energy_convergence.py — VMC energy convergence checker

Created 2026-06-15. Complements `tools/convergence_check.py` (which inspects
network weights); this one judges convergence from the *energy trajectory* in
`train_stats.csv` (cols `step,energy,ewmean,ewvar,pmove`).

**Method:** drop burn-in fraction → block-average energy into ~decorrelated
blocks → on a trailing window of blocks measure systematic DRIFT (linear-fit
slope × window) and test whether the slope is statistically flat (<2σ) →
report stochastic error bar (SEM of block means). Declares CONVERGED only when
drift ≤ `--tol` AND slope flat within noise. NaN-energy rows are skipped.

**Usage** (run in `ferminet-piku` conda env, see [[feedback-python-env]]):
```
python tools/energy_convergence.py <train_stats.csv | save_path_dir> \
    [--tol 1e-3] [--block 1000] [--burnin 0.2] [--window 20] [--plot out.png]
```
Accepts a [[job-save-paths]] directory directly (finds train_stats.csv).
Exit code 0 if converged, 1 otherwise.

**Interpreting verdicts:** the default `--tol 1e-3` is often ~5-6× the
stochastic SEM (~2e-4 E_h for these runs), so a run whose *slope is flat* but
drift > tol is really noise-limited/plateaued, not unconverged — the script
prints exactly this distinction ("NOT CONVERGED (by tol) but slope flat — run
is noise-limited"). A "still drifting (down)" verdict (slope NOT flat) means
genuine ongoing descent.

Example (2026-06-17, step ~312k): bc_relaxed pp q plateaued at
-90.5988±0.0002 (slope flat); unrelaxed pp q at -90.6480±0.0003 still
drifting down (-1.25e-7 E_h/step). Pairs with [[training-monitoring]].
