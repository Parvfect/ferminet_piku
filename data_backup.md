# Data Backup — Cluster Expiry (started 2026-08-17)

Tracking doc for backing up all important data off the expiring cluster
(`/home/u6em/parvfection.u6em` + `/projects/u6em/parv` both disappear).
Full plan: `.claude/plans/enchanted-inventing-flask.md`.

## Status legend
⬜ not started · 🟡 in progress · ✅ done · ⚠️ blocked/needs input

---

## Bucket A — Code + memory → GitHub
| Step | Status | Notes |
|---|---|---|
| Sync live memory dir → repo `claude_memory/` (40 files) | ⬜ | overwrites stale Jul-1 mirror |
| Commit docs (README + EXP-007/008) + memory on `muon_width` | ⬜ | |
| Push `muon_width` to origin | ⬜ | |
| Push `af` (1 commit ahead of origin/af) | ⬜ | avoid stranding committed work |
| Verify: `git log origin/muon_width -1`, clean status | ⬜ | |

## Bucket B — Curated run data (~11.7 GB)
Curated = newest checkpoint per run (67 dirs, 1.66 GB) + all `train_stats.csv`
(0.5 GB) + current inference `positions_*.npy` excl. `prior_*` (9.49 GB) + all
`srpd_*.txt` (439). Source: `/projects/u6em/parv`.

| Step | Status | Notes |
|---|---|---|
| Stage curated tree → `/projects/u6em/parv_backup_curated/` | ⬜ | `cp --parents` |
| Fold in Bucket C (QE decks + SLURM `.out` logs) | ⬜ | tiny |
| Write `MANIFEST.txt` (listing + du + sha256) | ⬜ | |
| Verify staged counts (67 ckpts, 65 csv, 21,802 positions) | ⬜ | |
| Archive → `parv_curated_backup_2026-08-17.tar.gz` | ⬜ | expect ~10-12 GB |

## Transfer + verify (needs SSH destination)
| Step | Status | Notes |
|---|---|---|
| SSH destination `user@host:/path` confirmed | ⚠️ | **awaiting user** |
| `rsync -avP` archive + MANIFEST → destination | ⬜ | |
| On destination: `sha256sum -c MANIFEST.txt` | ⬜ | zero mismatches |
| On destination: `numpy.load` one ckpt + one positions file | ⬜ | not truncated |
| Post-verify: delete on-cluster staging + archive | ⬜ | only after dest verified |

---

## Log
- 2026-08-17: Audit complete, plan approved, tracker created. Queue empty (no
  active jobs). `muon_width` already fully pushed; only uncommitted = 3 doc files.
