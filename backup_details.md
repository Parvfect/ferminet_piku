# Backup Details — muon-site FermiNet project (cluster expiry, 2026-08-17)

Complete record of **what** was backed up, **where** it lives, and **how to use
it** to resume runs and reproduce every result after the HPC cluster
(`/home/u6em/parvfection.u6em` + `/projects/u6em/parv`) is decommissioned.

Companion docs: `data_backup.md` (progress tracker),
`.claude/plans/enchanted-inventing-flask.md` (plan), `experiments/` (lab
notebook), `claude_memory/` (full research state, mirrored into git).

---

## 1. What this project is
Research fork of DeepMind FermiNet (JAX VMC) extended for **muon-site / µSR
studies in diamond and silicon supercells**. Central question: where does a muon
localise (tetrahedral **T-site** vs **bond-centre BC**) and does it form
**muonium** (a bound muon+electron = paramagnetic) vs stay **diamagnetic**
(screened H⁺). The muon is modelled as an extra heavy particle
(`cfg.system.particles/charges/masses`, muon = last particle, mass 206.768 mₑ);
solid-state runs use PBC + pseudopotentials, no pretraining.

**Headline scientific result (robust across the study):**
- **Diamond binds muonium at both T (1s-contact) and BC (bond-orbital).**
- **Silicon T-site is diamagnetic in every simulation** (unrelaxed, t_relaxed,
  t_seeded, classical) — a **host-material contrast**, robust to
  relaxation/seeding/classical-vs-quantum. (Silicon binds only *weakly* at BC.)
- The unseeded diamond BC muon localises **off-BC**; EXP-002 shows BC-seeding
  *holds* BC, and EXP-004/005 show the off-BC trap is an **optimisation** artefact
  (not sampling). See `experiments/` EXP-001…006 and `claude_memory/`.
- **Open frontier (planned, not run):** EXP-007/008 — bind silicon T-muonium
  (which real µSR sees but we miss) via a muon-anchored diffuse envelope +
  electron seeding.

---

## 2. Backup inventory — three locations

### (A) GitHub — all code, configs, jobs, tools, experiments, memory
Repo `git@github.com:Parvfect/ferminet_piku.git`. Pushed 2026-08-17:
- `muon_width` @ `fca72fa` (working branch; all research code + `claude_memory/`
  refreshed with the 39 live memory files + EXP-007/008 docs + these backup docs).
- `af` @ `68bb375` (older branch, pushed so nothing is stranded).
- `main` @ `9194f45` (upstream-tracking).
Everything needed to **rebuild the software and read the full research state** is
here. `requirements.txt` captures the `ferminet-piku` conda env.

### (B) Main curated data archive (~9.6 GB compressed, 12 GB raw)
On-cluster (transfer this off before expiry):
```
/projects/u6em/parv_curated_backup_2026-08-17.tar.gz
/projects/u6em/parv_curated_backup_2026-08-17.tar.gz.sha256
```
- **SHA256:** `7990832da80fe0d7dea707bb945f4f90047ce8f2d1867704dd344e047f7c7229`
- Unpacks to `parv_backup_curated/` mirroring paths under `/projects/u6em/parv`.
- **Contents** (22,409 files; integrity `gzip -t` = OK, 22,533 tar members incl. dirs):
  - **67 checkpoints** — the newest `qmcjax_ckpt_*.npz` of every run (~35 MB each;
    contains net weights + 4096 live MCMC walkers in `data/positions`).
  - **65 `train_stats.csv`** — full energy/variance trajectories (all runs).
  - **21,802 `positions_*.npy`** — current inference walker dumps (excludes
    `prior_*` re-run backups). These are the sampled configs SRPD/site analysis
    consumes.
  - **422 `srpd_*.txt`** — computed spin-resolved density curves.
  - **`_extra/`** — 32 QE `.in/.out` decks (EXP-001 PES + relaxations, from
    `~parvfect/qe/diamond_test`), 19 SLURM `.out` logs (incl. `muon_wide_burnin*`
    with step/energy history), and copies of `data_backup.md` + `BACKUP_PLAN.md`.
  - **`MANIFEST.txt`** at the archive root — per-file SHA256 + dir sizes.

### (C) Checkpoint supplement (~3.4 GB) — last-20 checkpoints of no-dump key runs
```
/projects/u6em/parv_ckpts_supplement_2026-08-17.tar.gz(.sha256)
```
- **SHA256:** `255081c66c54bc858c70009b4f2a8f1f844720170c898c9572da31e68795ae59`
- 3.1 GB compressed; unpacks to `parv_backup_ckpts_supplement/`. The **last 20 checkpoints** (120
files) of six scientifically-important runs that **never dumped inference
positions**, so their muonium/SRPD verdicts were computed by pooling walker
configs straight from checkpoints (see §5). Without these, re-analysis drops from
~82k configs (20×4096) to 4k (final ckpt only). Runs included:
`silicon_unpaired/t_seeded`, `diamond/unpaired/t_seeded/pp`,
`diamond/unpaired/classical/t_relaxed/pp`, `silicon_unpaired/classical/bc_relaxed`,
`diamond/unpaired/bc_relaxed/pp_wide_burnin_frozen`,
`diamond/unpaired/bc_relaxed/pp_wide_burnin_v3`. Has its own `MANIFEST.txt`.

---

## 3. Systems & weights — where each run is in the backup

Particles `(n↑, n↓, n_muon)`; quantum muon = doublet `(33,32,1)` (66 particles),
classical muon = fixed `'H'` atom `(33,32)` (65 e⁻, `mol.charge=1`). All PBC + PP
unless "nopp" (all-electron). Geometries are DFT-relaxed lattice coords.
"In backup" = which archive holds the run's data. **posdump** = inference
positions dumped (✓ ⇒ final ckpt + dump is enough; ✗ ⇒ its last-20 ckpts are in
the supplement (C)).

### Diamond (2×2×2, a = 6.74 bohr)
| Run | save_path (rel to /projects/u6em/parv) | particles | posdump | Result / verdict |
|---|---|---|---|---|
| #6 unrelaxed T, q | `diamond/unpaired/unrelaxed/pp` | 33,32,1 | ✓ (B) | **Muonium**, atomic/contact (g↑/g↓ 4.5–6.9), excess +0.42 |
| #7 classical T | `diamond/unpaired/classical/T_site/pp` | 33,32 | ✓ (B) | **Muonium** +0.523 @1.85 b (fixed-origin control, converged −90.696) |
| #4 classical BC | `diamond/unpaired/classical/bc` | 33,32 | ✓ (B) | **Bond-centred muonium** +0.53 @2.6 b (converged −90.730) |
| #5 bc_relaxed, q | `diamond/unpaired/bc_relaxed/pp` | 33,32,1 | ✓ (B) | Off-T trap; plateaued ~−90.598 |
| #8 bc_relaxed_2, q | `diamond/unpaired/bc_relaxed/pp_relax_2` | 33,32,1 | ✓ (B) | Off-T (not BC); −90.669 |
| #3 bc_relaxed, all-e⁻ | `diamond/unpaired/bc_relaxed/nopp` | 49,48,1 | ✓ (B) | T-site, diamagnetic (all-electron) |
| **#14 BC-seeded, q** | `diamond/unpaired/bc_relaxed/pp_bc_seeded` | 33,32,1 | ✓ (B) | **★ EXP-002 KEY: HOLDS BC, anisotropic muonium +0.47 @2.6 b; lowest-E quantum diamond ~−90.697** |
| #12 t_relaxed, q | `diamond/unpaired/t_relaxed/pp` | 33,32,1 | ✓ (B) | HOLDS expanded relaxed T-cage, **muonium**; −90.660 |
| #13 classical t_relaxed | `diamond/unpaired/classical/t_relaxed/pp` | 33,32 | ✗ (C) | relaxation-energy comparator (had a soft blow-up ~430k; latest ckpt clean) |
| #18 t_seeded, q | `diamond/unpaired/t_seeded/pp` | 33,32,1 | ✗ (C) | EXP-003b diamond analogue (seeded in relaxed T-cage) |
| #19 EXP-004 wide-burnin | `diamond/unpaired/bc_relaxed/pp_wide_burnin_v3` | 33,32,1 | ✗ (C) | wide muon proposal — off-BC re-collapse (H0) |
| EXP-005 frozen-adapter | `diamond/unpaired/bc_relaxed/pp_wide_burnin_frozen` | 33,32,1 | ✗ (C) | **0% BC even permanently wide → trap is optimisation artefact** |

### Silicon (2×2×2, a = 10.26 bohr)
| Run | save_path (rel) | particles | posdump | Result / verdict |
|---|---|---|---|---|
| #1 unrelaxed, q | `silicon_unpaired/unrelaxed` | 33,32,1 | ✓ (B) | T-site, **diamagnetic** (formation-dynamics control) |
| #2 classical T | `silicon_unpaired/classic` | 33,32 | ✓ (B) | **diamagnetic**, converged ~−62.92 |
| #9 bc_relaxed, q | `silicon_unpaired/bc_relaxed` | 33,32,1 | ✓ (B) | drifts to near-symmetric T cage, diamagnetic |
| **#15 BC-seeded, q** | `silicon_unpaired/bc_seeded` | 33,32,1 | ✓ (B) | **★ HOLDS BC, WEAK bond-centred muonium +0.25 (~½ diamond, diffuse ~3 b)** |
| #10 t_relaxed, q | `silicon_unpaired/t_relaxed` | 33,32,1 | ✓ (B) | diamagnetic; unseeded muon **avoided** relaxed cage (EXP-003 caveat) |
| #11 classical t_relaxed | `silicon_unpaired/classical/t_relaxed` | 33,32 | ✓ (B) | diamagnetic (fixed-H); NB training diverged >306k, latest *clean* ckpt = 306000 |
| **#17 t_seeded, q** | `silicon_unpaired/t_seeded` | 33,32,1 | ✗ (C) | **★ EXP-003b KEY: HOLDS relaxed cage yet still DIAMAGNETIC (never dumped positions → analysed from ckpts)** |
| #16 classical BC | `silicon_unpaired/classical/bc_relaxed` | 33,32 | ✗ (C) | fixed-H at Si BC (upper bound for BC contact density) |

### Other data present in the backup (final ckpt + any dumps, non-central)
Older 2×2 muon exploration and molecular/atomic FermiNet validation runs are also
captured (final checkpoint + inference dumps where they exist), but are **not**
core to the muon-site conclusions: `diamond/2x2*`, `diamond/1x1/pp`,
`carbon_lattice/*`, `silicon/`, `silicon/T_classical`, `ethyl_*`, `methyl*`,
`muonioum*`.

---

## 4. How to use the backup

**Restore + verify (on the destination machine):**
```bash
sha256sum -c parv_curated_backup_2026-08-17.tar.gz.sha256      # archive intact
tar -xzf parv_curated_backup_2026-08-17.tar.gz                 # -> parv_backup_curated/
cd parv_backup_curated && sha256sum -c MANIFEST.txt            # every file intact
tar -xzf ../parv_ckpts_supplement_2026-08-17.tar.gz            # supplement (C)
```

**Resume training** (if a compute env is rebuilt): point a config's
`cfg.log.save_path` / `restore_path` at the restored run dir; FermiNet restores
from the latest `qmcjax_ckpt_*.npz` automatically. Recreate the env from
`requirements.txt` (conda `ferminet-piku`). See `CLAUDE.md` for the run/inference
convention (custom PBC configs are executed as scripts, not the `ferminet` CLI).

**Reproduce analyses** (no cluster needed — CPU is fine):
- Energy convergence: `python tools/energy_convergence.py <run>/train_stats.csv`.
- Muon site / SRPD / muonium: `tools/muon_site_analysis.py` +
  `tools/srpd_extended_radius.py`. These are **checkpoint-aware**
  (`iter_frames`): with no `positions_*.npy` they read walkers from the last
  `n_ckpt` (default 20) checkpoints' `data/positions`. So the supplement (C)
  keeps the no-dump runs fully re-analysable. Case→geometry map + gotchas in
  `claude_memory/reference_tools.md`.

---

## 5. Why more than the final checkpoint was needed (rationale)
The muonium verdict is a **radius-resolved net-spin excess** `localized_excess(r)`
computed from many electron–muon configurations. Two sources supply those configs:
1. **Inference position dumps** (`inference/positions/positions_*.npy`, ~128k
   configs) — preserved in archive (B) for all posdump-✓ runs.
2. **Checkpoint walker state** (`data/positions`, 4096 configs each) — the
   analysis pools the **last ~20 checkpoints** for runs that never dumped
   positions (#17, #18, #13, #16, EXP-004/005). Keeping only the final checkpoint
   would cut those samples 20×. Hence supplement (C).

The **formation-dynamics** result (EXP-006: localization precedes binding; energy
⟂ local-spin decoupling) additionally swept ~19 checkpoints *across* each run's
trajectory. Those figures/numbers are recorded in
`claude_memory/project_bound_state_formation_dynamics.md` and `experiments/EXP-006`.
The runs it used (#6/#14/#15/#1) have inference dumps for their converged state, so
the **verdicts are preserved**; only the *across-training sweep* is not
independently re-runnable from this curated set (it needs the full checkpoint
trajectory, deliberately dropped to save space). If that reproduction is wanted
later, re-stage a coarse across-training sweep before the source is gone:
```bash
# example: 20 evenly-spaced ckpts across a run, add to a supplement
ls <run>/qmcjax_ckpt_*.npz | sort -t_ -k3 -n | awk 'NR%("'"$(($(ls <run>/qmcjax_ckpt_*.npz|wc -l)/20))"'")==0'
```

## 6. Excluded on purpose
- **All-but-final checkpoints of posdump-✓ runs** — redundant with their position
  dumps + `train_stats.csv` (this is the 272 GB → 12 GB reduction; 8,860 → 67+120
  checkpoints).
- **QE source build** `q-e-qe-7.4.1/` (~0.5 GB, reproducible) and QE wavefunction
  binaries — only the small `.in/.out` decks were kept.
- **Invalid EXP-004 runs** `pp_wide_burnin` / `_dense` / `_v2` (ran stale code,
  muon width never applied — superseded by `_v3`); final ckpt only.
- `prior_*` inference re-run backup dirs (superseded by current dumps).

---
_Generated 2026-08-17. Checksums and counts verified on-cluster; verify again on
the destination after transfer (`sha256sum -c`)._
