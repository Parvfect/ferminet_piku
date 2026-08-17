---
name: unpaired-electron-inference-sync
description: "Unpaired-electron (+1 up-spin) experiment active on branch af for all 6 production configs; inference configs synced to match on 2026-06-15"
metadata:
  node_type: memory
  type: project
  originSessionId: 875310ee-8c5f-490a-93e4-22279c5deca1
---

## Unpaired electron experiment (branch `af`)

As of 2026-06-15, all 6 production training configs (see [[training-monitoring]]
for the job table) were changed to add **one extra up-spin electron**
(`cfg.system.particles` first element +1: 32->33 or 48->49), and each was
given a new `cfg.log.save_path` under `/projects/u6em/parv/silicon_unpaired/...`
or `/projects/u6em/parv/diamond/unpaired/...`.

**Why:** Studying the effect of an unpaired electron (net spin) on the
muon/defect system, vs the original spin-balanced setups.

## Inference config sync (2026-06-15)

The corresponding inference configs had NOT been updated and still pointed at
the old (32/32 or 48/48) particle counts and old `*2x2_muon*`/`*silicon*`
save paths (which hold OLD paired checkpoints — restoring those into the new
particle-count network would crash on shape mismatch, since `train.py`
checks `cfg.log.save_path` first when restoring).

Edited all 6 inference configs to match their training counterparts exactly:

| Training config | Inference config | particles | save_path (both train & inference now) | geometry change needed? |
|---|---|---|---|---|
| `silicon/pp.py` | `silicon/inference.py` | (32,32,1)->(33,32,1) | `/projects/u6em/parv/silicon_unpaired/unrelaxed` | no (already matched) |
| `silicon/t_site.py` | `silicon/t_site_inference.py` | (32,32)->(33,32) | `/projects/u6em/parv/silicon_unpaired/classic` | no |
| `diamond/bc_relaxed/nopp.py` | `diamond/bc_relaxed/inference_nopp.py` | (48,48,1)->(49,48,1) | `/projects/u6em/parv/diamond/unpaired/bc_relaxed/nopp` | **yes** — replaced old rounded C coords with precise DFT-relaxed coords from `nopp.py` |
| `diamond/bc_relaxed/pp_classic.py` | `diamond/bc_relaxed/inference_pp_T.py` | (32,32)->(33,32) | `/projects/u6em/parv/diamond/unpaired/classical/bc` | no (already had precise coords incl. H at T-site) |
| `diamond/bc_relaxed/pp.py` | `diamond/bc_relaxed/inference_pp.py` | (32,32,1)->(33,32,1) | `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp` | **yes** — same precise-coord replacement as nopp |
| `diamond/pp.py` (unrelaxed) | `diamond/test_run_inference.py` | (32,32,1)->(33,32,1) | `/projects/u6em/parv/diamond/unpaired/unrelaxed/pp` | no (already matched ideal positions) |

For each pair, `mol.charge`, `cfg.system.charges`, `cfg.system.masses`, and
all other physics settings were already consistent and untouched.

**save_path change was not explicitly requested** but was necessary: it's
what makes `train.py`'s restore logic pick up the new (33,32,*)/(49,48,*)
checkpoints instead of crashing on the old (32,32,*)/(48,48,*) ones.

**Verified on disk** (2026-06-15): all 6 new save_path directories exist and
contain active `qmcjax_ckpt_*.npz` sequences (latest ranging ~ckpt_096000 for
silicon configs up to ~ckpt_566000 for diamond bc_relaxed nopp), confirming
the unpaired training runs are writing there and inference will restore
correctly.

## Loose end (resolved 2026-06-15)

`ferminet/jobs/diamond_2x2/bc_relaxed/pp_T_inf.sh` previously had a stale
`mkdir -p /projects/u6em/parv/diamond/2x2_muon/bc_relaxed/pp_T/positions`
(old path). This has been fixed — the script now creates
`/projects/u6em/parv/diamond/unpaired/classical/bc/inference/positions`,
matching `cfg.log.save_path`. See [[current-status]] for the first
`pp_T_inf` run using this corrected script.

**How to apply:** When the unpaired-electron experiment concludes or if
particle counts change again, re-check all 6 inference configs against their
training counterparts using this table as the reference for what "matching"
looks like.
