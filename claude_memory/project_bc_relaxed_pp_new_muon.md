---
name: bc-relaxed-pp-new-muon-result
description: Inference result for diamond bc_relaxed pp_new — muon localised at T-site not BC site
metadata: 
  node_type: memory
  type: project
  originSessionId: ce11a168-063b-4f75-b5e7-29cd39b4d40d
---

## bc_relaxed pp_new inference (2026-06-05)

**Finding:** The quantum muon localised at a **T-site**, not the BC site as intended.

- Inference config: `ferminet/configs/diamond/bc_relaxed/inference_pp_new.py`
- Loaded checkpoint: `qmcjax_ckpt_188000.npz`
- Save path: `/projects/u6em/parv/diamond/2x2_muon/bc_relaxed/pp_new/positions/` (1000 files, 1100 steps)
- Positions shape: `(4 devices, 128 batch, 65 particles, 3)` — particle index 64 is the muon

**Muon localization:**
- Fractional coord std ~0.04 → muon IS localized (not diffusing)
- Mean fractional coords (fcc supercell): (0.755, 0.233, 0.765)
- Mean Cartesian: (~10.25, 6.66, 6.73) bohr

**Site identification:**
- 4 nearest C atoms at 2.77, 2.86, 2.93, 2.95 bohr → matches **T-site** (ideal: 4×C at 2.92 bohr)
- Closest C-C bond midpoint (BC site) is **2.74 bohr away** — far too large for BC
- BC site would require 2 nearest C at ~1.46 bohr (half bond length ~2.95/2)

**Likely cause:** The C atoms are DFT-relaxed around the BC site, but the network found the T-site lower in energy. The mismatched relaxation doesn't confine the muon to BC.

**Why:** Intended to study the muon at the BC site in diamond (bc_relaxed geometry). Result needs checking — may indicate a network training issue or genuine physics.

**How to apply:** When re-running or analysing bc_relaxed pp_new, note the muon is at the T-site. Any SRPD or observable analysis should account for this. Consider whether to retrain from a BC-site-constrained initialisation.
