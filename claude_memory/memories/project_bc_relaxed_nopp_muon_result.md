---
name: project_bc_relaxed_nopp_muon_result
description: bc_relaxed nopp (all-electron) quantum muon at T-site; RMS 0.38 bohr; SRPD DIAMAGNETIC (not muonium) — contrast with PP runs
metadata: 
  node_type: memory
  type: project
  originSessionId: c23dff7a-dcfd-4e6e-9c22-8b60e2f50003
---

All-electron (no-PP) bc_relaxed quantum muon inference. Config `diamond/bc_relaxed/inference_nopp.py`, particles=(49,48,1)=98 (muon last, idx 97), a=6.74, positions at `/projects/u6em/parv/diamond/unpaired/bc_relaxed/nopp/inference/positions` (1000 files, 512k samples).

- **Site:** localizes at the **T-site**, NOT the BC site. Histogram peak cubic-frac ≈ [1.40,1.20,0.75]; 4 roughly equidistant C neighbours ~2.2–2.7 bohr; ~4.6–4.8 bohr from intended BC site (mid C0–C1). Same outcome as every other diamond run.
- **Spread:** RMS radius 0.380 bohr (per-axis 0.24/0.22/0.20, ~isotropic); concentration R≈0.99. Tighter than PP bc_relaxed (~0.60 bohr).
- **SRPD:** **unpolarized / DIAMAGNETIC at contact — NOT muonium.** up/dn pair ratio ≈1.0 across all r windows (r<0.1: 446 vs 448). This contrasts with [[project_bc_relaxed_pp_muon_result]] and [[project_unrelaxed_pp_muon_result]] which showed muonium (strong single-spin contact). Possible cause: core electrons screen contact density in all-electron, or nopp convergence/sampling difference — unresolved.
- **Artifact warning:** raw g(r) shows a fake spike in bin 0 (g_down=3.73) — it's a SINGLE electron-muon pair count divided by the vanishing r→0 shell volume (5.2e-7 bohr³). Ignore the first ~2 bins of any SRPD; counts only meaningful from ~bin 4+.

To run: extended `tools/muon_site_analysis.py` with case `bc_relaxed_nopp` (now supports per-case n_particles/n_up/n_dn). `python tools/muon_site_analysis.py bc_relaxed_nopp [site|spread|srpd]`.
