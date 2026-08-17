---
name: unrelaxed-pp-muon-result
description: "Inference result for diamond UNRELAXED pp (unpaired-electron, 33,32,1) — muon localised at ideal T-site"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23883b38-2982-49b8-9517-0fb762f321bc
---

## unrelaxed diamond pp inference (analysed 2026-06-15)

**Finding:** Quantum muon localised tightly at an **ideal tetrahedral (T)
interstitial site** — cubic-frac (0.25, 0.25, 0.75), Cartesian
(8.43, 8.43, 11.80) bohr. All 4 nearest C at exactly 2.919 bohr (= one C-C
bond length), the textbook T-site. ~2.79 bohr from nearest BC site.

- Config: `ferminet/configs/diamond/test_run_inference.py` (particles
  (33,32,1), ideal diamond geometry, a=6.74)
- Positions: `/projects/u6em/parv/diamond/unpaired/unrelaxed/pp/inference/positions/`
  (1000 files, 512k muon samples)
- Localisation: R≈0.97 per axis; 98% within 1.0 bohr of peak, 100% within
  2.0 bohr, median 0.50 bohr.

**All three diamond quantum-muon runs now agree → muon prefers T-site:**
unrelaxed (this), bc_relaxed pp [[bc-relaxed-pp-muon-result]], bc_relaxed
pp_new [[bc-relaxed-pp-new-muon-result]]. Relaxing C around the bond centre
does NOT pull the muon to BC.

**Analysis script (reviewable, reusable):**
`tools/muon_site_analysis.py` — `python tools/muon_site_analysis.py <case> <mode>`,
cases `unrelaxed` / `bc_relaxed_pp` / `silicon`, modes `site` / `spread` /
`srpd`. Folds positions into fcc primitive cell, uses circular mean +
3D-histogram peak (Cartesian mean is meaningless due to periodic wrap-around),
classifies BC vs T by nearest-carbon pattern. `srpd` replicates
`observables.py:cal_spin_resolved_pair_density` (electron-muon g(r), per-spin,
shell-volume normalised; r_search=0 -> rounding min-image). See [[silicon-quantum-muon-result]].

## RESULTS (validated 2026-06-15)

**Muon spread (zero-point localisation width), unrelaxed diamond T-site:**
- per-axis RMS (0.32, 0.32, 0.33) bohr — ISOTROPIC (spherical cage)
- overall RMS radius sqrt(<r^2>) = 0.556 bohr (~0.29 Å); median |r| 0.50 bohr;
  90% within 0.80 bohr, 99% within 1.06 bohr. Tightly localised.

**SRPD g(r) electron-muon pair density, unrelaxed diamond:**
- STRONG spin-up excess at contact: g_up ~0.29 vs g_down ~0.03 at r~0.08 bohr
  (~10:1). Converges (g_up≈g_down) by ~2-3 bohr.
- # electrons within 1 bohr of muon: up=0.38, down=0.12 (total ~0.50, mostly up).
- Physics: MUONIUM-like state — muon binds the extra up-spin electron
  (particles (33,32,1) has 1 extra up-e), giving net spin-up contact density =
  the hyperfine-relevant electron spin density. Diamond muon HAS a local moment.
- Contrast with silicon (no spin polarisation) — see [[silicon-quantum-muon-result]].
- Config SRPD default rmax=1 bohr (only contact shell); bump cfg.observables.srpd.rmax
  for full g(r). g(r) tables written to /tmp/srpd_<case>_rmax<R>.csv.
