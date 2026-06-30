---
name: potential-bugs
description: "Known bugs found while monitoring training, deliberately deferred until a safe point to fix"
metadata: 
  node_type: memory
  type: project
  originSessionId: 70601531-6df8-419a-80eb-fefbe0cf7f7a
---

## Silicon unrelaxed quantum-muon: permanently NaN `ewvar` (exp. variance)

**Bug:** At step 30446 of the silicon quantum-muon unrelaxed run
(`ferminet/jobs/silicon/muon.sh`, save path
`/projects/u6em/parv/silicon_unpaired/unrelaxed/`), the per-step loss hit
`-inf` for two consecutive steps. `ferminet/train.py:1086` only guards the
`weighted_stats` EMA update with `not jnp.isnan(loss)`, which doesn't catch
`±inf`. The `-inf` observations poisoned the EMA recurrence (`mean_new =
mean_old + alpha*(obs - mean_old)`), making `weighted_stats.mean`/`.variance`
permanently `nan`. Since `weighted_stats` is checkpointed, this is baked into
every checkpoint from `qmcjax_ckpt_030446` onward (latest as of 2026-06-13:
`qmcjax_ckpt_060952.npz`).

**Confirmed impact:** Cosmetic only — `ewvar`/`ewmean` columns in
`train_stats.csv` are `nan` from step 30446 onward, but the actual per-step
`energy` (loss) is unaffected and correct (e.g. step 60952:
`energy=-62.772`, `ewvar=nan`). Training, optimizer, and gradients are NOT
affected — `weighted_stats.variance` is logging-only.

**Why not fixed yet:** Decided 2026-06-13 to defer. Fixing requires (a)
editing `train.py` — shared by all 6 production configs, all currently
mid-run — and (b) patching the live checkpoint's `weighted_stats` field on
shared storage. Both considered too risky while runs are progressing well;
"everything works fine, train now, fix later."

**Fix when ready:**
1. `ferminet/train.py:1086` — change `if not jnp.isnan(loss):` to
   `if jnp.isfinite(loss):` (prevents recurrence for all configs going
   forward).
2. One-time patch of the latest `qmcjax_ckpt_*.npz` in
   `/projects/u6em/parv/silicon_unpaired/unrelaxed/`: back up, set the
   `weighted_stats` field to `None` (0-d object array), re-save other fields
   unchanged. On next resume, the EMA reinitializes cleanly from the first
   finite loss (`WeightedStats(mean=loss, variance=0.0)`).

**How to apply:** When investigating other "variance=nan" issues during
[[training-monitoring]], check for `±inf` energy spikes (`grep "inf E_h"` in
the `.out` log, excluding `nan E_h`) as the signature of this specific bug —
distinct from the harmless transient `nan E_h` + reset cycles seen in all 6
configs. Apply the fix above only when the user explicitly decides it's a
safe point to interrupt/restart the silicon job.
