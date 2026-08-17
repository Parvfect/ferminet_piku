---
name: job-save-paths
description: "Lookup: cfg.log.save_path (project dir holding train_stats.csv + checkpoints) for each of the 7 production training jobs"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd7babd3-df02-4561-aef6-eb933702fa94
---

## save_path / project dir per production job

Each save_path holds `train_stats.csv` (header
`step,energy,ewmean,ewvar,pmove`) and a `checkpoints/`-less flat dir of
`qmcjax_ckpt_*.npz` + the csv. Pulled from each config's
`cfg.log.save_path`. Use for validity checks without re-reading configs.
Pairs with [[training-monitoring]] (job scripts / logs / job names).

| # | Job (script) | Config | save_path |
|---|---|---|---|
| 1 | `silicon/muon.sh` | `configs/silicon/pp.py` | `/projects/u6em/parv/silicon_unpaired/unrelaxed` |
| 2 | `silicon/t_site.sh` | `configs/silicon/t_site.py` | `/projects/u6em/parv/silicon_unpaired/classic` |
| 3 | `diamond_2x2/bc_relaxed/nopp.sh` | `configs/diamond/bc_relaxed/nopp.py` | `/projects/u6em/parv/diamond/unpaired/bc_relaxed/nopp` |
| 4 | `diamond_2x2/bc_relaxed/pp_classic.sh` | `configs/diamond/bc_relaxed/pp_classic.py` | `/projects/u6em/parv/diamond/unpaired/classical/bc` |
| 5 | `diamond_2x2/bc_relaxed/pp.sh` | `configs/diamond/bc_relaxed/pp.py` | `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp` |
| 6 | `diamond_2x2/muon.sh` | `configs/diamond/pp.py` | `/projects/u6em/parv/diamond/unpaired/unrelaxed/pp` |
| 7 | `diamond_2x2/classical_muon/t_site_pp.sh` | `configs/diamond/classical_muon/t_site_pp.py` | `/projects/u6em/parv/diamond/unpaired/classical/T_site/pp` |
| 10 | `silicon/t_relaxed.sh` | `configs/silicon/t_relaxed.py` | `/projects/u6em/parv/silicon_unpaired/t_relaxed` |
| 11 | `silicon/t_relaxed_classical.sh` | `configs/silicon/t_relaxed_classical.py` | `/projects/u6em/parv/silicon_unpaired/classical/t_relaxed` |
| 12 | `diamond_2x2/t_relaxed/pp.sh` | `configs/diamond/t_relaxed/pp.py` | `/projects/u6em/parv/diamond/unpaired/t_relaxed/pp` |
| 13 | `diamond_2x2/t_relaxed/classical.sh` | `configs/diamond/t_relaxed/classical.py` | `/projects/u6em/parv/diamond/unpaired/classical/t_relaxed/pp` |
| 14 | `diamond_2x2/bc_relaxed/bc_seeded.sh` | `configs/diamond/bc_relaxed/bc_seeded.py` | `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_bc_seeded` |
| 15 | `silicon/bc_seeded.sh` | `configs/silicon/bc_seeded.py` | `/projects/u6em/parv/silicon_unpaired/bc_seeded` |
| 16 | `silicon/bc_relaxed_classical.sh` | `configs/silicon/bc_relaxed_classical.py` | `/projects/u6em/parv/silicon_unpaired/classical/bc_relaxed` |
| 17 | `silicon/t_seeded.sh` | `configs/silicon/t_seeded.py` | `/projects/u6em/parv/silicon_unpaired/t_seeded` |
| 18 | `diamond_2x2/t_relaxed/t_seeded.sh` | `configs/diamond/t_relaxed/t_seeded.py` | `/projects/u6em/parv/diamond/unpaired/t_seeded/pp` (NOT STARTED — created at launch) |

Note (2026-06-22): jobs 10–13 are the new **T-relaxed** runs (DFT-relaxed
geometry around the T-site muon). All four save dirs created empty 2026-06-22, no
collisions. Beware the empty leftover capital-`T_relaxed` dirs
(`silicon_unpaired/T_relaxed`, `diamond/unpaired/T_relaxed`, both Jun 19) — the
new configs use lowercase `t_relaxed` and are unrelated to those.

Notes:
- Job 4's script `--output=pp_T.out` and `--job-name=bc_diamond_pp_T`
  (naming artifact — it runs `pp_classic.py`, save_path `.../classical/bc`).
- Checkpoints + csv are written flat in the save_path (no `checkpoints/`
  subdir for these runs); each save_path also has an `inference/` subdir from
  the paired inference runs.
- train_stats.csv flushes at checkpoint boundaries (~every 2000 steps), so
  the csv normally lags the `.out` log's last step by up to ~2000 steps; a
  larger or mtime-stale gap means the run hung/died after its last
  checkpoint (e.g. job 4 hung at ~step 237521 on 2026-06-15, csv frozen at
  step 236000).
