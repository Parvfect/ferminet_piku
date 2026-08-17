---
name: project-diamond-results
description: Diamond PP-run GSE comparison table and muon-site inference results; use whenever asked about diamond energies or muon sites
metadata: 
  node_type: memory
  type: project
  originSessionId: b3788c45-2950-4eff-a0df-720fbed90f1a
---

## Diamond PP runs — ground-state energy comparison

★ = ACTIVE (still training). Plateau values = trailing-window mean ± 1 SEM from `tools/energy_convergence.py`. See [[project-job-save-paths]] for paths. See [[feedback-conventions]] for ★ rule.

| Run | Muon | Geometry | Step | Plateau E (E_h) | Status |
|---|---|---|---|---|---|
| #4 classical BC | fixed @ BC | bc_relaxed | 338k (frozen) | **−90.73010 ± 0.00029** | CONVERGED (frozen) |
| #7 classical T-site | fixed @ T | unrelaxed | 472k (frozen) | **−90.69624 ± 0.00015** | CONVERGED (frozen) |
| #8 bc_rel_2 | quantum | bc_relaxed (charged-state, expanded BC) | 280k (frozen) | **−90.66904 ± 0.00029** | STOPPED @280k (muon at off-T trap) |
| #6 unrelaxed | quantum | unrelaxed | 522k (frozen) | **−90.66312 ± 0.00017** | CONVERGED (frozen) |
| #14 bc_seeded ★ | quantum | bc_relaxed (EXP-002) | 218k | **−90.66541 ± 0.00045** | RUNNING, −3.5e-7/step; ~3.6 mHa from #8 |
| #12 t_relaxed ★ | quantum | t_relaxed | 302k | **−90.64346 ± 0.00042** | RUNNING, −1.4e-6/step |
| #13 classical t_rel ★ | fixed @ T | t_relaxed | 146k | **−90.64169 ± 0.00050** | RUNNING, −1.2e-6/step |
| #5 bc OLD | quantum | bc_relaxed (original) | 336k (frozen) | **−90.59794 ± 0.00151** | CLOSED (frozen) |

**Interpretation:** Classical (#4, #7) NOT directly comparable to quantum (#6, #8, #14) — classical fixes muon as point charge, no ZPM → lower energy. Among quantum runs: at matched steps #8 (bc_rel_2) was consistently 10–28 mHa BELOW #6 (unrelaxed) but both now roughly equal at large step count. Do NOT cite #6 < #8 until #8 is fully converged to ~500k steps.

**How to refresh:** Don't re-run frozen rows. Rerun only ★ active rows: `python tools/energy_convergence.py <save_path>` in `ferminet-piku` conda env.

---

## Diamond muon-site inference results

### T-site results (all unseeded runs)
| Run | Site | Dist to BC | Spread RMS | Spin character |
|---|---|---|---|---|
| #6 unrelaxed (q) | T (0.25,0.25,0.75) | ~2.79 bohr | 0.556 bohr, isotropic | **MUONIUM** g↑:g↓ ~10:1 contact |
| #5 bc_relaxed pp (q) | T (0.5,0.55,0.5) | 7.1 bohr | 0.598 bohr | Weak muonium, ~1.3:1 (different T-site) |
| #8 bc_rel_2 (q) | off-T (0.32 bohr from T, one C at ~2.0 bohr) | 4.06 bohr | 0.54 bohr | DIAMAGNETIC contact (but init trap, not true site) |
| #3 bc_relaxed nopp (q, all-electron) | T | ~4.6–4.8 bohr from BC | 0.380 bohr (tighter, AE) | DIAMAGNETIC (contact; AE screens contact) |
| #12 t_relaxed (q) | T (expanded cage) | — | 0.66 bohr | **MUONIUM** g↑:g↓ ~6:1, loc_excess ~+0.4 |

### BC-seeded results (EXP-002)
| Run | Site | Dist to BC | Anisotropy | Spin character |
|---|---|---|---|---|
| #14 bc_seeded (q, 56k net) | BC | 0.30 bohr | Oblate, par/perp 0.48 (bond-pinned) | contact looks diamagnetic (screening cloud) — but extended-radius: **ANISOTROPIC MUONIUM** |
| #14 bc_seeded (q, 176k net) | BC | **0.037 bohr** | RMS 0.61 bohr, ~isotropic | Same: contact ≈ diamagnetic; loc_excess **+0.47 @ ~2.7 bohr** = bond-centred muonium |
| #4 classical BC (fixed, 222k→338k) | BC (clamped) | 0 (fixed) | — | contact ≈ diamagnetic; loc_excess **+0.53 @ ~2.6 bohr** = bond-centred muonium |

**KEY PHYSICS — BC is anisotropic/bond-centred muonium, NOT diamagnetic (corrected 2026-06-30):**
Contact SRPD (rmax=1) shows g↑≈g↓ inside 1 bohr — that's just the +1 symmetric screening cloud. The REAL signature is at bond scale (~2.5–3 bohr): `tools/srpd_extended_radius.py` shows localized_excess = +0.47–0.53 e⁻, 5-7× above uniform-band line. Both quantum (ZPM) and classical (clamped muon) show this → it's an **electronic-structure property of BC**, not a ZPM artifact. Diamond has muonium at BOTH T (1s-contact, spherical) AND BC (bond-orbital, anisotropic) — shape differs, not presence.

**T-site physical picture:** expanded cage (+0.017 bohr, +0.6% C outward), diamond muon holds the relaxed cage. Contrast silicon: contracted cage (−0.049 bohr, −1.1%), muon FLEES 8.47 bohr to an unrelaxed T-site. Expansion holds, contraction expels.

**Analysis:** `python tools/muon_site_analysis.py <case> {site|spread|srpd}` — cases: `unrelaxed`, `bc_relaxed_pp`, `bc_relaxed_pp_relax_2`, `bc_seeded`, `bc_relaxed_nopp`, `diamond_t_relaxed`, `diamond_classical_bc`. `tools/srpd_extended_radius.py <case> <rmax> <nbins> <stride>` for extended net-spin (handles fixed-origin/classical cases too).

**site_report heuristic in muon_site_analysis.py is BUGGY** — tests 1st→2nd nearest-C gap (mislabels symmetric BC as "T"). Trust explicit distances, not the printed label.
