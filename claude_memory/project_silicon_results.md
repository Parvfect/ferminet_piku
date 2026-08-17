---
name: project-silicon-results
description: Silicon PP-run GSE comparison table and muon-site inference results; use whenever asked about silicon energies or muon sites
metadata: 
  node_type: memory
  type: project
  originSessionId: b3788c45-2950-4eff-a0df-720fbed90f1a
---

## Silicon PP runs — ground-state energy comparison

★ = ACTIVE (still training). See [[feedback-conventions]] for ★ rule.

| Run | Muon | Geometry | Step | Plateau E (E_h) | Status |
|---|---|---|---|---|---|
| #2 classical T-site | fixed @ T | unrelaxed | 224k (frozen) | **−62.91337 ± 0.00015** | FINISHED @224k (frozen) |
| #11 classical t_rel ★ | fixed @ T | t_relaxed | 172k | **−62.91564 ± 0.00024** | RUNNING −1.8e-7/step; ~2 mHa BELOW frozen #2 |
| #9 bc_relaxed | quantum | bc_relaxed | 138k (frozen) | **−62.86382 ± 0.00012** | STOPPED @138k (muon drifted to T cage) |
| #10 t_relaxed | quantum | t_relaxed | 138k (frozen) | **−62.87133 ± 0.00048** | STOPPED @130k (muon avoided relaxed cage, EXP-003) |
| #1 unrelaxed | quantum | unrelaxed | 180k (frozen) | **−62.86151 ± 0.00047** | CLOSED @180k (frozen) |
| #15 bc_seeded ★ | quantum | bc_relaxed (EXP-002) | 54k | **−62.82833 ± 0.00032** | RUNNING −3.2e-6/step; RECOVERED after 35–40k blowup |
| #17 t_seeded ★ | quantum | t_relaxed (EXP-003b) | 26k | **−62.70285 ± 0.00042** | RUNNING −1.6e-5/step; fresh-net climbing |

**Interpretation:** Classical vs quantum NOT directly comparable (ZPM difference). Among quantum runs: #9 bc_relaxed and #1 unrelaxed converge to essentially the same energy (~0.2 mHa) at matched steps — silicon geometry makes little difference to GSE. #15/#17 too early for geometry verdict.

**How to refresh:** Don't re-run frozen rows. Rerun ★ active rows. NOTE: early-spike VMC outliers in warm-up phases blow up block-avg (don't cite tool output during warm-up; use live log estimate instead).

**#15 regression history:** descended to −62.758@33k, blew up at 35–40k, stuck at ~−62.12 for 35k steps. Fixed by truncating train_stats.csv to step≤34000, moving post-35k ckpts to `post35k_regression_bak/`, restarting from ckpt_034000 (job 5426315). Now recovered and descending cleanly.

---

## Silicon muon-site inference results

### All silicon runs: DIAMAGNETIC (no muonium)

| Run | Site | Spread | Spin character |
|---|---|---|---|
| #1 unrelaxed (q) | T-like, 4 Si at 4.17–4.76 bohr | 0.844 bohr, anisotropic (0.41,0.62,0.40) | Diamagnetic: g↑≈g↓, up=0.19/dn=0.20 within 1 bohr |
| #2 classical T-site | T (0.75a)³, fixed-H | — | Diamagnetic: contact g↑~0.15, g↓~0.18; SAME as quantum #1 |
| #9 bc_relaxed (q, unseeded) | T cage, 4.4–4.7 bohr from BC | 0.865 bohr, loose | Diamagnetic; extended loc_excess ≈0 to 6 bohr (no bound state) |
| #10 t_relaxed (q, unseeded) | Unrelaxed T-site (coordinating Si: 0.0002 bohr moved) | 0.81 bohr | Diamagnetic (H0 confirmed) — muon AVOIDED relaxed cage |
| #17 t_seeded (q, SEEDED) | Relaxed T-site (7.695³ bohr, HOLDS cage) | 0.78 bohr, isotropic (0.45³) | Diamagnetic (EXP-003b, H0) — g↑≈g↓, localized_excess ≈0/slightly −ve, +0.08 @5.7 bohr, no plateau |
| #11 classical t_relaxed | fixed H @ relaxed T (7.695³) | — | Diamagnetic (EXP-003b upper bound) — contact ∫≤1 bohr 0.192 up / 0.199 dn; localized_excess ≈0 |

**#10 EXP-003 key finding:** muon sat in an UNRELAXED T-site (Si #15,11,14,10 moved 0.0002 bohr vs design Si #4,5,6,7 moved 0.0487 bohr), 8.47 bohr from the intended relaxed T-site. P(at relaxed site) ≈ 0. Why: DFT contracted the Si cage (−1.1%); quantum muon prefers roomier unrelaxed cage. → Motivates #17 T-seeded (EXP-003b).

**Silicon vs diamond contrast:**
- Diamond T-site: strong contact muonium g↑:g↓ ~10:1 (wide-gap, open cage, binds e⁻)
- Silicon T-site: no spin polarization g↑≈g↓ (narrower gap, different screening)
- Silicon BC: same diamagnetic character (no localized net spin — unlike diamond BC which has +0.47–0.53 e⁻ localized)

**Extended-radius check (silicon bc_relaxed #9):** run `tools/srpd_extended_radius.py silicon_bc_relaxed 6.0 120 4` — spin-symmetric screening cloud (g/bulk ~8× contact, crosses bulk ~1.5 bohr) but localized_excess ≈0/slightly negative to 6 bohr → no bound state at any scale.

**Analysis:** `python tools/muon_site_analysis.py <case> {site|spread|srpd}` — cases: `silicon` (unrelaxed), `silicon_classical` (classical T), `silicon_bc_relaxed`, `silicon_t_relaxed` (EXP-003), `silicon_t_seeded` (EXP-003b, quantum), `silicon_classical_t_relaxed` (EXP-003b, fixed-H).

**#17 t_seeded / #11 classical t_relaxed — no inference dump needed:** #17's paired inference never wrote `positions_*.npy` (inference/positions empty). The tools now read walker positions straight from **training checkpoints** — a new `checkpoints=` key + `muon_site_analysis.iter_frames()` (falls back to `data/positions` from the last `n_ckpt=20` npz when no positions files exist; each ckpt = batch_size 4096 full 66-particle configs). `srpd_extended_radius.py` shares `iter_frames`. So bound-state analysis runs off checkpoints for ANY quantum run without a fresh inference SLURM job. #11 classical DID have a 2026-07-01 inference dump (its training later diverged past 330k = `inf`, but those frames predate the blowup, finite). See [[project-experiments]] EXP-003b.

**FCC primitive vectors for silicon (a=10.26 bohr):** `[[a,a,0],[0,a,a],[a,0,a]]`. BC site at `(0.125a,0.125a,0.125a) = 1.2825 bohr`. T-site at `(0.75a)³ = 7.695 bohr`.
