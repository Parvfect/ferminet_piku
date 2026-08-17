---
name: env-kfac-jax-fresh-net-crash
description: "The parvfection HOME conda env's kfac_jax build crashes fresh-net multi-host KFAC runs (NamedSharding assert); the parvfect OLD env has a different 0.0.8 build that works. Use the old env for fresh-net SLURM runs until home env re-tested."
metadata: 
  node_type: memory
  type: project
  originSessionId: b5294807-1976-4391-8fc1-3c3c38362e3a
---

## Two ferminet-piku conda envs behave differently on fresh-net multi-host KFAC (2026-07-07)

There are TWO `ferminet-piku` conda envs on this cluster:
- **OLD** `/home/u6em/parvfect.u6em/miniforge3` (the pre-migration `parvfect` account home)
- **HOME** `/home/u6em/parvfection.u6em/miniforge3` (current `parvfection` account home)

Both report **identical package versions**: jax 0.6.2, jaxlib 0.6.2, jax-cuda12-plugin/pjrt 0.6.2,
kfac_jax **0.0.8**, folx 0.2.20, optax 0.2.8. BUT the `kfac_jax` *code* differs — they are different
git builds both tagged `0.0.8.dist-info` (nearly every file in the package differs). The HOME build is
newer/dev (imports `unsafe_get_axis_names_DO_NOT_USE`, "JAX v0.10.0 or newer" branch); the OLD build is
older (`core.axis_frame`, `jax_version >= (0,4,36)` guards).

### Symptom
A **fresh-net** 8-node run in the HOME env completes burn-in (2000 steps) then **crashes on the FIRST
KFAC optimizer step** (`train.py:1086` → `step` train.py:286 → `kfac_jax optimizer.step:1428` →
`_verify_args_and_get_step_counter:741` → `get_first(step_counter)`), with
`AssertionError: assert isinstance(value.sharding, jax.NamedSharding)` in
`kfac_jax/_src/utils/parallel.py`. Job FAILS exit 1:0 in ~6 min, no checkpoint written.
**Warm-restore runs do NOT hit this** (their opt_state/step_counter come from checkpoint), which is why
the long-running production jobs (all warm restores) never showed it. Every working fresh-net run
(#18 t_seeded, the earlier wide_burnin dense/v2) used the **OLD env**.

### Root cause & fix
The env split, NOT the ferminet code, NOT the EXP-004 `muon_move_width` change (burn-in uses mcmc, the
crash is in KFAC internals on step_counter). Fixes applied 2026-07-07:
1. **Immediate:** point SLURM scripts at the OLD env for fresh-net runs. `wide_burnin.sh` had been
   switched to HOME by branch `muon_width` commit `3d7b294`; reverted its `source ...activate` line back
   to `/home/u6em/parvfect.u6em/miniforge3`. This is what all working `.sh` (e.g. `bc_seeded.sh`,
   `silicon/t_seeded.sh`) use.
2. **Home-env repair (DONE + VERIFIED):** replaced HOME env's `kfac_jax` package + `kfac_jax-0.0.8.dist-info`
   with the OLD env's working copy (backed up originals as `kfac_jax.bak_20260707` /
   `kfac_jax-0.0.8.dist-info.bak_20260707` under HOME site-packages). **CONFIRMED end-to-end 2026-07-07 on an
   8-node fresh-net run (job 5529283, `pp_wide_burnin_v3`):** cleared the first KFAC step, wrote
   `ckpt_000000` (muon width 0.3 / electrons 0.02 per `tools/muon_width_check.py --particles 33,32,1`),
   trained past step 0 normally. The `NamedSharding` crash is gone.

### CRITICAL: the OLD env CANNOT run our edited ferminet
The OLD `parvfect` env imports a **stale/copied `ferminet`** (NOT the editable repo) — an old-env fresh-net
run of `wide_burnin.py` showed **NO `[EXP-004]` print and NO `Initial MCMC width` line**, i.e. it runs a
`train.py` without the `muon_move_width` block at all. So the old env would silently run EXP-004 WITHOUT
the width intervention. **Only the HOME env imports our editable repo `train.py`** (proven: its runs print
`[EXP-004] muon_move_width applied ... 0.3`). ⇒ EXP-004 (and any run needing our fork edits) MUST use the
HOME env. The old env is only usable for runs that don't depend on repo-side train.py changes.

### Practical rule (UPDATED)
Use the **HOME `parvfection` env** for all our runs — it imports the editable repo AND (post-fix) has the
working kfac_jax. The old `parvfect` env runs a stale ferminet copy and must NOT be used for anything
depending on our code. If HOME env packages ever get reinstalled/upgraded, re-verify kfac_jax on an 8-node
fresh-net run (the crash only shows on fresh nets + multi-host; warm restores hide it). See [[current-status]].
