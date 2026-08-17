---
name: project-account-migration-conda-fix
description: "Root cause + fix for the 2026-07-01 mass job-death event — new account's home has no miniforge3, job scripts sourced ~/miniforge3 which silently resolved to nothing"
metadata: 
  node_type: memory
  type: project
  originSessionId: 248c52b7-c847-454b-9e95-3038f4e68812
---

## Root cause of the 2026-07-01 mass job death (previously attributed to a "cluster/node event")

Account migrated from `parvfect.u6em` (old) to `parvfection.u6em` (new, current). All 6 active
job scripts (`silicon/t_relaxed_classical.sh`, `diamond_2x2/t_relaxed/{pp,classical}.sh`,
`diamond_2x2/bc_relaxed/bc_seeded.sh`, `silicon/{bc_seeded,t_seeded}.sh`) do:
```
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet-piku
```
`~` resolves via `$HOME`, now `/home/u6em/parvfection.u6em` — but miniforge3 (with the
`ferminet-piku` env: jax 0.6.2 + jax-cuda12-plugin + kfac-jax 0.0.8 + folx + ferminet -e install)
was only ever installed under the OLD account's home, `/home/u6em/parvfect.u6em/miniforge3`.
No `set -e` in the scripts, so `source`/`conda activate` failures were silent, `python` fell
back to whatever's on PATH (no jax/ferminet), and the run crashed near-instantly — matching the
"exit 6:0 task/node abort, NOT clean TIMEOUT" pattern for all 6 jobs clustered 08:23–09:20 UTC
2026-07-01, right after admin cancelled 2 inference jobs at 06:56 for "account migration
housekeeping." Far more likely explanation than a node-hardware event.

**Old account's miniforge3 IS accessible from the new account** (dirs 755, env binaries 755/777,
group `brics.u6em` has read+execute) — confirmed by directly sourcing it and running
`python -c "import jax"` successfully from the new account's session (login-node CPU only;
GPU untested there, no GPU on login nodes).

## Fix applied (2026-07-01, fast path)

Edited all 6 active job scripts: `source ~/miniforge3/bin/activate` →
`source /home/u6em/parvfect.u6em/miniforge3/bin/activate` (absolute, old-account path). Also
deleted the dead `cd ferminet_remote` line in each (that dir doesn't exist under the new
account — under the OLD account the repo lived at `~/ferminet_remote/ferminet_piku`, hence
the two-`cd` pattern; under the new account it's directly `~/ferminet_piku`). Verified the
full `cd ~ → activate → conda activate → cd ferminet_piku → import ferminet,jax,kfac_jax,folx`
sequence end-to-end on the login node.

**Caveat:** this fix depends on the old account's home surviving. If/when `parvfect.u6em` is
fully decommissioned as part of the migration, this breaks again.

## Durable env DONE (2026-07-01), but NOT yet cut over — old-account env stays authoritative for active runs

Built fresh at `/home/u6em/parvfection.u6em/miniforge3`, env `ferminet-piku`, python 3.12.
**First attempt failed**: this cluster is **aarch64** (Isambard Grace-Hopper ARM, confirmed via
`uname -m`), not x86_64 — the generic `Miniforge3-Linux-x86_64.sh` installer's bundled
micromamba gave `Exec format error`. Fixed by using `Miniforge3-Linux-aarch64.sh`.

**Package versions are NOT a match for the old env** — user supplied the correct known-good
recipe (exact order matters, kfac-jax must be reinstalled from git AFTER the initial `-e`
install pulls in a default one):
```
pip install -e /home/u6em/parvfection.u6em/ferminet_piku
pip install matplotlib
pip uninstall -y kfac-jax
pip install --no-cache-dir git+https://github.com/google-deepmind/kfac-jax@e57a7c023b595d44440183746a382b2abc02ff2c
pip install -U "jax[cuda12]==0.4.30"
pip install chex==0.1.89
pip install distrax==0.1.5
pip install optax==0.2.4
```
Result: **jax 0.4.30** (not 0.6.2 like the old/active env), kfac-jax from the pinned commit
(not PyPI 0.0.8), distrax 0.1.5, optax 0.2.4, chex 0.1.89. All imports verified OK
(`ferminet, jax, kfac_jax, folx, optax, chex, distrax, pyscf`). Full freeze saved to
`/home/u6em/parvfection.u6em/miniforge3/envs/ferminet-piku/new_env_freeze.txt`. **Not yet
GPU-tested** — login node has no GPU (`cuInit` → `CUDA_ERROR_NO_DEVICE` there, expected).

**DECISION (user, 2026-07-01): do NOT cut the 6 active job scripts over to this new env yet.**
The active checkpoints/optimizer state were produced under jax 0.6.2 (old-account env) —
mixing a different jax/optax/kfac-jax version into a resumed run risks silent
incompatibility (checkpoint format, optimizer state shape, RNG behavior). The 6 scripts stay
on the old-account absolute path (`/home/u6em/parvfect.u6em/miniforge3`) for now. The new
aarch64 env is for **future fresh runs / inference only**, not for resuming these checkpoints,
until explicitly decided otherwise.

## PATH_MIGRATION.md (repo root, untracked) is STALE — do not run it

That doc assumes `/projects/u6em/parv/` is unwritable by the new account and proposes copying
all data to `/projects/u6em/parvfection/` + sed-rewriting all 36 config `save_path`/
`restore_path` values. **Verified false**: a full recursive ownership sweep of `/projects/u6em/parv/`
found zero files not owned by `parvfection.u6em` — everything (checkpoints, train_stats.csv for
both active and completed runs) is already owned by the new account and fully writable
(confirmed via a live write test). Running that doc's sed command would break all configs by
pointing them at a directory that doesn't exist. Recommend deleting `PATH_MIGRATION.md` or at
minimum not executing it. See [[job-save-paths]] for the (still-correct, unchanged) real paths.
