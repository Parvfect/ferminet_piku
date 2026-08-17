---
name: silicon-quantum-muon-result
description: "Inference result for silicon UNRELAXED quantum muon (33,32,1) — T-like site, anisotropic spread, NO spin polarisation (diamagnetic, unlike diamond muonium)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23883b38-2982-49b8-9517-0fb762f321bc
---

## Silicon unrelaxed quantum-muon inference (analysed 2026-06-15)

- Config: `ferminet/configs/silicon/inference.py` (particles (33,32,1),
  ideal Si diamond geometry, a=10.26 bohr, PP). 66 particles, muon idx 65.
- Positions: `/projects/u6em/parv/silicon_unpaired/unrelaxed/inference/positions/`
  (1000 files, 512k samples)
- Analysed with `tools/muon_site_analysis.py silicon {site,spread,srpd}`.

**Site:** localised at a TETRAHEDRAL-like interstitial, NOT the BC site.
Histogram peak cubic-frac ~(0, 0.45, 0); 4 nearest Si at ~4.17-4.76 bohr
(Si-Si bond 4.44 bohr → T-site coordination; BC would be 2 Si at ~2.22 bohr).
~7 bohr from intended BC site. Concentration R≈0.97-0.98.

**Spread (zero-point width):** MORE diffuse and ANISOTROPIC vs diamond.
- per-axis RMS (0.41, 0.62, 0.40) bohr — elongated along one axis (anisotropic
  cage; diamond was isotropic 0.32 each).
- overall RMS radius 0.844 bohr; median |r| 0.75 bohr; only 75% within 1.0 bohr
  (vs 98% for diamond), 100% within 2.0 bohr. Bigger lattice → bigger cage.

**SRPD g(r) electron-muon pair density — KEY CONTRAST WITH DIAMOND:**
- g_up ≈ g_down at ALL r (contact: g_up~0.15, g_down~0.17). NO spin
  polarisation at the muon.
- # electrons within 1 bohr: up=0.19, down=0.20 (total ~0.39, unpolarised;
  diamond was ~0.50 heavily up).
- Physics: silicon muon is DIAMAGNETIC-like — does NOT bind a localised
  spin-polarised electron. The extra up-spin electron (particles (33,32,1))
  stays delocalised, unlike diamond where it binds to form muonium with a net
  spin-up contact density. Consistent with Si being narrow-gap vs wide-gap
  diamond.

**CLASSICAL silicon SRPD (validated 2026-06-15) — matches quantum:**
- Config `ferminet/configs/silicon/t_site_inference.py`: classical muon = fixed
  H at T-site (0.75a,0.75a,0.75a)=(7.695,7.695,7.695) bohr, particles (33,32)
  (65 electrons, NO muon particle), mol.charge=1, srpd.use_fixed_origin=True.
  Positions `/projects/u6em/parv/silicon_unpaired/classic/inference/positions/`
  shape (4,128,195)=65 particles. Run via `tools/muon_site_analysis.py
  silicon_classical srpd` (fixed-origin branch: rvec = electron - origin).
- Result: contact g_up~0.15, g_down~0.18 (unpolarised, slightly more DOWN);
  electrons within 1 bohr up=0.19/down=0.20; within 7 bohr up=20.8/down=20.1.
- NEARLY IDENTICAL to the quantum silicon muon (up=0.19/0.20 within 1 bohr).
  => quantum (0.84 bohr ZP spread) and classical (fixed H) treatments give the
  same local electron density; both DIAMAGNETIC, no muonium. Validates T-site
  as the classical position.

Cross-refs: diamond muonium result [[unrelaxed-pp-muon-result]]; all diamond
runs at T-site [[bc-relaxed-pp-muon-result]] [[bc-relaxed-pp-new-muon-result]].
