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
| Sync live memory dir → repo `claude_memory/` (39 files) | ✅ | flat layout replaced stale `memories/` subdir |
| Commit docs (README + EXP-007/008) + memory on `muon_width` | ✅ | commit fca72fa |
| Push `muon_width` to origin | ✅ | origin/muon_width @ fca72fa |
| Push `af` (1 commit ahead of origin/af) | ✅ | origin/af @ 68bb375 |
| Verify: `git log origin/muon_width -1`, clean status | ✅ | working tree clean, in sync |

## Bucket B — Curated run data (~11.7 GB)
Curated = newest checkpoint per run (67 dirs, 1.66 GB) + all `train_stats.csv`
(0.5 GB) + current inference `positions_*.npy` excl. `prior_*` (9.49 GB) + all
`srpd_*.txt` (439). Source: `/projects/u6em/parv`.

| Step | Status | Notes |
|---|---|---|
| Stage curated tree → `/projects/u6em/parv_backup_curated/` | ✅ | rsync, 22,356 files, ~3 min |
| Fold in Bucket C (QE decks + SLURM `.out` logs) | ✅ | `_extra/`: 32 QE decks (24K) + 19 logs (161M) + plan/tracker |
| Write `MANIFEST.txt` (listing + du + sha256) | ✅ | 22,409 files checksummed |
| Verify staged counts (67 ckpts, 65 csv, 21,802 positions) | ✅ | all exact; 422 srpd; total 12 GB |
| Archive → `parv_curated_backup_2026-08-17.tar.gz` | ✅ | 9.6 GB, gzip -t OK, 22,533 members, sha256 `7990832d…` |

## Bucket B2 — Checkpoint supplement (last-20 ckpts, no-dump key runs)
Added after reviewing experiments: 6 runs (#17, #18, #13, #16, EXP-004/005) that
never dumped inference positions — their muonium verdicts pool the last ~20
checkpoints. See `backup_details.md` §3/§5.

| Step | Status | Notes |
|---|---|---|
| Stage last-20 ckpts of 6 no-dump runs | ✅ | 120 files, 3.4 GB |
| Archive → `parv_ckpts_supplement_2026-08-17.tar.gz` + MANIFEST | ✅ | 3.1 GB, sha256 `255081c6…` |
| `backup_details.md` (systems/weights/where) written | ✅ | committed with tracker |

## Transfer (user pulling locally)
| Step | Status | Notes |
|---|---|---|
| Pull main archive + supplement + `.sha256` to local | ⬜ | user-driven |
| `sha256sum -c` both on destination | ⬜ | main `7990832d…`, supp `255081c6…` |
| Post-verify cleanup on-cluster (staging + archives) | ⬜ | only after local verify |

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
