---
name: feedback-conventions
description: "All feedback from user about how to work — ★ for active runs, block-averaged energies, afterany follow-ups, energy comparison tables always included, python env"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b3788c45-2950-4eff-a0df-720fbed90f1a
---

## ★ on active runs in tables

In run/energy comparison tables, mark ACTIVE (still-training) runs with ★ next to the run name; leave frozen/closed/not-started rows unmarked.

**Why:** user explicitly asked for this convention — makes "which numbers are still moving / are upper bounds" instantly scannable.

**How to apply:** Always add ★ to the Run column for RUNNING/descending rows when producing [[project-diamond-results]] or [[project-silicon-results]] tables, or any run-status table. Add caption note "★ = active".

---

## Energy values: always use block-averaged train_stats.csv, NEVER log instantaneous prints

**Why:** Per-step `Step N: … E_h` log prints are INSTANTANEOUS, single-sample ±0.03 E_h noise. Comparing two raw endpoints gave a completely wrong ΔE for #14 (looked +0.035 UP while the block-avg run was flat at ~−90.649).

**How to apply:**
- Step RANGE (span): use `grep -oE 'Step [0-9]+:' <log> | sed -n '1p;$p'` — log prints only
- Energy VALUES: always `train_stats.csv` block-averages (1000-step blocks, mean ± SEM); cross-check ewmean column
- Flag fresh-net warm-up garbage explicitly (E can be −38/−23 E_h in first few hundred steps)
- Flag low-confidence for runs with only ~hundreds of steps since restart

---

## Energy comparison tables included every daily check (both diamond AND silicon)

**Why:** user confirmed (2026-06-25) they want BOTH [[project-diamond-results]] and [[project-silicon-results]] tables EVERY time they ask for training status, without asking separately.

**How to apply:** Always include both tables. Both MUST include a Step (iteration) column — current step for active runs, final/frozen step for stopped ones. Refresh only active rows; reuse frozen/converged rows.

---

## afterany not afterok for SLURM follow-ups

**Why:** jobs end via TIMEOUT or CANCELLED, never COMPLETED → `afterok` would never fire.

**How to apply:** Always `--dependency=afterany:<jobid>` for follow-up submissions.

---

## Python environment: `ferminet-piku` conda env

Any `python tools/...` command must run in the `ferminet-piku` conda env.

**How to apply:** When running energy_convergence.py, muon_site_analysis.py, srpd_extended_radius.py, or any repo tool, activate `ferminet-piku` first.

---

## QE (Quantum ESPRESSO): `module load libfabric` then `pw.x`

**Why:** QE pw.x lives in `~/qe/diamond_test/`. Run from that dir with `module load libfabric`.

---

## Parallel sbatch submits: use absolute paths in subshells

**Why:** `cd relative/path && sbatch` changes the persistent shell cwd, breaking sibling parallel calls.

**How to apply:** `(cd /abs/path && sbatch <script>)` for each submit in parallel batches.

---

## Don't cite #6 unrelaxed < #8 bc_rel_2 as settled

**Why:** Final-plateau table compares converged #6 (522k) to stopped #8 (280k) — premature. On matched-step basis #8 was consistently 10–28 mHa LOWER and still descending.

**How to apply:** When discussing diamond quantum-run energy ordering, always state it's not settled until #8 reaches comparable step count (~500k+). See [[project-diamond-results]].
