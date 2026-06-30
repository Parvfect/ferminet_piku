---
name: bc-relaxed-pp-muon-result
description: "Inference result for diamond bc_relaxed pp (unpaired-electron, particles 33,32,1) — muon localised at T-site not BC site"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23883b38-2982-49b8-9517-0fb762f321bc
---

## bc_relaxed pp inference (analysed 2026-06-15)

**Finding:** The quantum muon localised tightly at a **T-site (tetrahedral
interstitial)**, NOT the intended BC site, despite the C atoms being
DFT-relaxed around the bond center.

- Inference config: `ferminet/configs/diamond/bc_relaxed/inference_pp.py`
  (unpaired-electron variant: particles=(33,32,1), 66 particles total, muon =
  flattened index 65 → reshape (..., 66, 3)[:, 65, :])
- Positions dir: `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp/inference/positions/`
  (1000 files, each `(4,128,198)` = 198/3 = 66 particles; 512k muon samples)

**Localization (tight):** fcc-fractional circular concentration R≈0.97 each
dim; 96% of samples within 1.0 bohr of density peak, 100% within 2.0 bohr,
median 0.53 bohr. NOTE: naive Cartesian mean is meaningless here (std ~4.3
bohr from periodic wrap-around) — must use circular stats / fold into
primitive cell.

**Site:** peak at Cartesian ≈ (10.1, 10.4, 10.1) bohr ≈ cubic-frac
(0.5, 0.55, 0.5) mod 1. Nearest 4 C atoms at ~2.74, 2.74, 3.08, 3.12 bohr →
tetrahedral T-site. BC site (midpoint of relaxed C0-C1, cubic-frac
(0.125,0.125,0.125)) is **~7.1 bohr away**; a true BC site would have 2
carbons at ~2.04 bohr (half the 4.08-bohr relaxed C0-C1 bond).

**Same outcome as [[bc-relaxed-pp-new-muon-result]]** (pp_new also T-site).
Both BC-relaxed PP quantum-muon runs prefer the T-site over the relaxed BC
site → consistent with the muon naturally favouring the T-site when not
confined during training. To get a BC result likely need a BC-constrained
init / MCMC seeded at the bond center.

Analysis method (lattice fcc primitive [[a,a,0],[0,a,a],[a,0,a]], a=6.74):
min-image distances via f=d@inv(L); f-=round(f); d=f@L. Localization via
circular mean of 2πf + 3D histogram peak. Script: `tools/muon_site_analysis.py
bc_relaxed_pp {site,spread,srpd}`.

## RESULTS (validated 2026-06-15) — spread + SRPD

**Does NOT match the relaxed BC geometry** — muon escaped to a T-site
(see above), 7.1 bohr from where the C atoms were relaxed.

**Spread:** RMS radius 0.598 bohr, ~isotropic (0.35, 0.36, 0.32) — essentially
identical to unrelaxed diamond T-site (0.56 bohr) → same kind of tetrahedral cage.

**SRPD g(r) — WEAKER muonium than unrelaxed diamond:**
- contact g_up:g_down ≈ 1.3:1 (g_up~0.23, g_down~0.18 at smallest bin) — only
  mild spin-up excess, vs unrelaxed diamond's ~5:1 ([[unrelaxed-pp-muon-result]])
  and silicon's 1:1 ([[silicon-quantum-muon-result]]). Intermediate /
  near-diamagnetic. Muonium local moment largely suppressed.
- CAVEAT: unrelaxed muon at T-site (0.25,0.25,0.75) vs bc_relaxed at
  (0.5,0.55,0.5) — DIFFERENT tetrahedral interstitials, so the SRPD difference
  may partly reflect a different local environment, not purely the relaxation.
