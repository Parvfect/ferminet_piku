#!/usr/bin/env python3
"""Check whether a FermiNet VMC run has converged in *energy*.

Complements tools/convergence_check.py (which inspects network weights). For
VMC the meaningful convergence signal is the energy trajectory in
`train_stats.csv` (columns: step,energy,ewmean,ewvar,pmove).

Why not just compare consecutive iterations? The per-step energy is dominated
by MCMC/stochastic noise, so |E_{n+1}-E_n| never shrinks to zero even at full
convergence. Instead we:

  1. drop an initial burn-in fraction,
  2. block-average the energy into ~decorrelated blocks,
  3. on a trailing window of blocks, measure the systematic DRIFT (linear-fit
     slope x window) and test whether the slope is statistically flat,
  4. report the stochastic error bar (SEM of block means) -- a tolerance
     tighter than this is physically meaningless.

A run is declared CONVERGED when the trailing-window drift is below `--tol`
AND the slope is within ~2 sigma of zero (i.e. flat within the noise).

Usage:
    python tools/energy_convergence.py <train_stats.csv | run_dir> \
        [--tol 1e-3] [--block 1000] [--burnin 0.2] [--window 20] [--plot out.png]

`run_dir` may be a save_path directory containing train_stats.csv.
"""
import os
import sys
import argparse
import numpy as np


def load_stats(path):
    """Load train_stats.csv; accept a file or a directory containing it."""
    if os.path.isdir(path):
        path = os.path.join(path, "train_stats.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no train_stats.csv at {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    step = np.asarray(data["step"], dtype=float)
    energy = np.asarray(data["energy"], dtype=float)
    # Keep only rows with a finite energy (energy can NaN on isolated steps).
    good = np.isfinite(energy)
    return step[good], energy[good], path, int((~good).sum())


def block_average(energy, block):
    """Return block means over non-overlapping windows of size `block`."""
    nblocks = len(energy) // block
    if nblocks < 2:
        raise ValueError(
            f"need >=2 blocks; have {len(energy)} samples / block={block} "
            f"= {nblocks}. Lower --block.")
    trimmed = energy[: nblocks * block].reshape(nblocks, block)
    return trimmed.mean(axis=1), nblocks


def analyse(path, tol, block, burnin, window):
    step, energy, resolved, n_nan = load_stats(path)
    n_total = len(energy)
    if n_total < 10:
        raise ValueError(f"too few finite samples ({n_total}) to assess")

    # 1. Burn-in.
    start = int(burnin * n_total)
    e_eq = energy[start:]
    s_eq = step[start:]

    # 2. Block averaging.
    block_means, nblocks = block_average(e_eq, block)
    # Step number at the centre of each block (for slope-per-step + plotting).
    block_steps = s_eq[: nblocks * block].reshape(nblocks, block).mean(axis=1)

    # 3. Trailing window of blocks.
    w = min(window, nblocks)
    if w < 3:
        raise ValueError(
            f"trailing window has only {w} blocks; need >=3. Lower --block "
            f"or --burnin, or collect more steps.")
    yb = block_means[-w:]
    xb = block_steps[-w:]

    mean_E = yb.mean()
    sem = yb.std(ddof=1) / np.sqrt(w)          # stochastic error of the plateau

    # Linear drift across the trailing window (slope in E_h per step).
    A = np.vstack([xb - xb.mean(), np.ones(w)]).T
    (slope, _), residuals, *_ = np.linalg.lstsq(A, yb, rcond=None)
    # Standard error of the slope from the residual scatter.
    yhat = A @ np.array([slope, yb.mean()])
    dof = max(w - 2, 1)
    resid_var = np.sum((yb - yhat) ** 2) / dof
    sxx = np.sum((xb - xb.mean()) ** 2)
    slope_err = np.sqrt(resid_var / sxx) if sxx > 0 else np.inf

    span = xb[-1] - xb[0]
    drift = slope * span                        # total systematic change
    drift_err = slope_err * span

    # Block-to-block changes (your "difference between successive iterations",
    # but at block resolution where it is actually meaningful).
    dblk = np.abs(np.diff(yb))
    max_step = dblk.max()
    mean_step = dblk.mean()

    # 4. Verdict.
    drift_below_tol = abs(drift) <= tol
    slope_flat = abs(slope) <= 2 * slope_err     # indistinguishable from zero
    converged = drift_below_tol and slope_flat

    print("=" * 72)
    print(f"file: {resolved}")
    print(f"samples: {n_total} finite"
          + (f" ({n_nan} NaN rows skipped)" if n_nan else ""))
    print(f"burn-in: dropped first {start} ({burnin:.0%}); "
          f"blocks: {nblocks} of {block} steps; trailing window: {w} blocks")
    print(f"step range analysed: {int(s_eq[0])} .. {int(step[-1])}")
    print("-" * 72)
    print(f"plateau energy (trailing mean): {mean_E:.5f} +/- {sem:.5f} E_h  (1 SEM)")
    print(f"systematic drift over window:   {drift:+.2e} +/- {drift_err:.1e} E_h "
          f"(over {int(span)} steps)")
    print(f"slope:                          {slope:+.2e} +/- {slope_err:.1e} E_h/step")
    print(f"block-to-block |dE|:            mean {mean_step:.2e}, max {max_step:.2e} E_h")
    print("-" * 72)
    print(f"tolerance: {tol:.1e} E_h")
    if tol < sem:
        print(f"  ! WARNING: tol ({tol:.1e}) is below the stochastic error bar "
              f"({sem:.1e}); convergence at this level is not resolvable. "
              f"Consider tol >~ {sem:.0e} or a larger --block/run.")
    print(f"  drift <= tol?            {drift_below_tol}  "
          f"(|{drift:.2e}| vs {tol:.1e})")
    print(f"  slope flat (<2 sigma)?   {slope_flat}  "
          f"(|{slope:.2e}| vs 2x{slope_err:.1e})")
    print("=" * 72)
    if converged:
        print("VERDICT: CONVERGED — energy plateaued within tolerance and flat "
              "within noise.")
    elif slope_flat and not drift_below_tol:
        print("VERDICT: NOT CONVERGED (by tol) but slope is statistically flat; "
              "the run is noise-limited — loosen --tol toward the SEM.")
    else:
        print(f"VERDICT: NOT CONVERGED — energy still drifting "
              f"({'down' if slope < 0 else 'up'}).")
    print("=" * 72)

    return dict(block_steps=block_steps, block_means=block_means,
                mean_E=mean_E, sem=sem, slope=slope, converged=converged)


def plot(res, out, tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bs, bm = res["block_steps"], res["block_means"]
    plt.figure(figsize=(10, 5))
    plt.plot(bs, bm, ".-", ms=3, lw=0.8, label="block-mean energy")
    plt.axhline(res["mean_E"], color="g", ls="--", label="plateau mean")
    plt.fill_between([bs[0], bs[-1]], res["mean_E"] - tol, res["mean_E"] + tol,
                     color="g", alpha=0.15, label=f"+/- tol ({tol:.0e})")
    plt.xlabel("training step")
    plt.ylabel("energy (E_h)")
    plt.title(f"Energy convergence — "
              f"{'CONVERGED' if res['converged'] else 'not converged'}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"wrote plot -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="train_stats.csv or a run/save_path directory")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="energy drift tolerance in E_h (default 1e-3)")
    p.add_argument("--block", type=int, default=1000,
                   help="steps per averaging block (default 1000)")
    p.add_argument("--burnin", type=float, default=0.2,
                   help="fraction of run to discard as burn-in (default 0.2)")
    p.add_argument("--window", type=int, default=20,
                   help="number of trailing blocks for the plateau test (default 20)")
    p.add_argument("--plot", type=str, default=None,
                   help="optional output PNG of the block-mean trajectory")
    args = p.parse_args()

    res = analyse(args.path, args.tol, args.block, args.burnin, args.window)
    if args.plot:
        plot(res, args.plot, args.tol)
    sys.exit(0 if res["converged"] else 1)


if __name__ == "__main__":
    main()
