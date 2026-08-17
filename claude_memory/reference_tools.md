---
name: reference-tools
description: "Reference for repo analysis tools — energy_convergence.py, muon_site_analysis.py, srpd_extended_radius.py — usage and interpretation"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b3788c45-2950-4eff-a0df-720fbed90f1a
---

## tools/energy_convergence.py

Assesses VMC energy convergence from `train_stats.csv` via block-averaging + trailing-window drift test. Use in `ferminet-piku` conda env.

```
python tools/energy_convergence.py <train_stats.csv | save_path_dir> \
    [--tol 1e-3] [--block 1000] [--burnin 0.2] [--window 20] [--plot out.png]
```

Accepts a [[project-job-save-paths]] directory directly. Exit code 0 if converged.

**Verdicts:** default `--tol 1e-3` ≈ 5-6× stochastic SEM (~2e-4 E_h); "NOT CONVERGED (by tol) but slope flat" = noise-limited/plateaued; "still drifting (down)" = genuine ongoing descent. For runs with VMC early-spike outliers (warm-up phases), the tool gives garbage — use robust recent estimate from log instead.

## tools/muon_site_analysis.py

Position-inference analysis: site location, spread, contact SRPD. Run from repo root.

```
python tools/muon_site_analysis.py <case> {site|spread|srpd [rmax [nbins]]}
```

**Cases (as of 2026-06-30):**
- `unrelaxed` — diamond unrelaxed quantum (33,32,1), a=6.74
- `bc_relaxed_pp` — diamond bc_relaxed pp (33,32,1), a=6.74
- `bc_relaxed_pp_relax_2` — diamond bc_relaxed pp_relax_2 (C_BC_RELAXED_2 geom, bc_pair (0,1))
- `bc_seeded` — diamond bc_seeded quantum (C_BC_RELAXED_2, bc_pair (0,1), positions in pp_bc_seeded/inference/positions)
- `bc_relaxed_nopp` — diamond bc_relaxed all-electron (49,48,1), custom n_particles/n_up/n_dn
- `diamond_t_relaxed` — diamond T-relaxed quantum (C_T_RELAXED geom, a=6.74, bc_pair (0,8))
- `diamond_classical_bc` — classical fixed-BC (33,32), fixed-origin branch, origin = H/BC position
- `silicon` — silicon unrelaxed quantum (33,32,1), a=10.26
- `silicon_classical` — silicon classical T-site (33,32), fixed-origin
- `silicon_bc_relaxed` — silicon bc_relaxed quantum (SI_BC_RELAXED geom, bc_pair (0,8), a=10.26)
- `silicon_t_relaxed` — silicon T-relaxed quantum (SI_T_RELAXED geom, a=10.26, bc_pair (0,8))
- `silicon_t_seeded` — silicon T-SEEDED quantum (EXP-003b; reads from CHECKPOINTS, not inference dump)
- `silicon_classical_t_relaxed` — silicon classical fixed-H @ relaxed T (EXP-003b; n_particles=65, fixed_origin)

**Checkpoint-aware loading (added 2026-07-08):** `muon_site_analysis.iter_frames(cfg)` yields `(W, n_particles, 3)` frames from either `positions_*.npy` OR, if a case has `checkpoints=<save_path>` (+ optional `n_ckpt`, default 20) and no positions files, from the last n_ckpt training checkpoints' `data/positions` (each = batch_size 4096 full N-particle walker configs). `srpd_extended_radius.py` shares it. So the muonium/bound-state analysis runs off checkpoints for ANY quantum run without a fresh inference SLURM job — no positions dump needed. `load_muons` now takes `cfg` (not a dir).

**KNOWN BUG:** `site_report` heuristic tests 1st→2nd nearest-host gap → mislabels symmetric BC as "T" and 1-close-host off-T as "BC". Trust explicit distances printed, NOT the site label.

**Analysis method:** FCC primitive cell `[[a,a,0],[0,a,a],[a,0,a]]`, min-image via `f=d@inv(L); f-=round(f); d=f@L`. Circular mean of 2πf + 3D-histogram peak. Positions = reshape(positions, (-1, n_particles, 3))[:, muon_idx, :] (muon = LAST particle, 0-indexed).

## tools/srpd_extended_radius.py

Extended-radius SRPD to measure net spin (muonium discriminator) out beyond 1 bohr. Handles fixed-origin (classical) and quantum cases. Use in `ferminet-piku` env.

```
python tools/srpd_extended_radius.py <case> <rmax> <nbins> <stride>
```

Example: `python tools/srpd_extended_radius.py bc_seeded 6.0 120 2` (quantum 176k net)  
Example: `python tools/srpd_extended_radius.py diamond_classical_bc 4.5 90 2` (classical fixed-BC)

**Key output — `localized_excess`:** cumulative N↑−N↓ minus the uniform-band line (lone doublet up-electron spread evenly ∝r³). Interpretation:
- `localized_excess` → +0.3 to +1.0 and PLATEAUS within lattice scale = bound electron = **MUONIUM**
- `localized_excess` ≈ 0 or tracks 0 at all radii = delocalized = **DIAMAGNETIC**
- For BC: localized_excess peaks at ~+0.47 (quantum) / ~+0.53 (classical) at ~2.6–2.7 bohr = bond-centred muonium

**Important:** r≈0.01 first bin is a shell-volume artifact (single pair count / vanishing volume). Ignore first ~2–4 bins.

## tools/muon_site_analysis.py srpd rmax

Contact SRPD (rmax ≤ 1 bohr default) measures g(r) inside 1 bohr only. For BC site this LOOKS diamagnetic (symmetric screening cloud). Must use `srpd_extended_radius.py` to detect bond-centred muonium.
