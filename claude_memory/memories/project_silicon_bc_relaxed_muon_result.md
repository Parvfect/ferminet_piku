---
name: project-silicon-bc-relaxed-muon-result
description: "Silicon bc_relaxed (UNSEEDED) quantum muon does NOT hold BC — drifts ~4.4 bohr to a near-symmetric T cage, diamagnetic, loose spread. Different site from diamond's off-T. Motivates silicon bc_seeded (#15)."
metadata:
  node_type: memory
  type: project
  originSessionId: bc-seeded-analysis
---

## Silicon bc_relaxed (unseeded) quantum muon — does NOT hold BC, drifts to a T cage

Analysed 2026-06-27 (branch `af`). Inference dumped 1000 `positions_*.npy` (512k
samples) in `/projects/u6em/parv/silicon_unpaired/bc_relaxed/inference/positions`.
Added case **`silicon_bc_relaxed`** to `tools/muon_site_analysis.py`
(SI_BC_RELAXED geometry from `configs/silicon/inference_bc_relaxed.py`,
bc_pair (0,8), a=10.26, quantum muon = last of 66 particles).

### Result: the muon left the engineered BC and sits at a near-symmetric T cage
- **4.4–4.7 bohr from the intended BC** (= 1.09 bond lengths away).
- Coordination fingerprint (4 nearest Si / host bond 4.35 bohr): **0.96, 0.96,
  1.06, 1.10** — 2+2, all ~1 bond, NONE close → a **near-symmetric tetrahedral (T)
  interstitial cage**, not BC (BC would be 2 Si at 0.5 bond) and not the diamond
  off-T.
- **Spread loose & anisotropic:** RMS radius 0.865 bohr (per-axis 0.46/0.60/0.43),
  median |r| 0.78, 99% within 1.64 bohr. ~1.7× looser than the diamond bc_seeded BC
  muon (0.51 bohr) — sits in an open cage (Si a=10.26 ≈ 1.5× diamond).
- **SRPD: DIAMAGNETIC, low pair density** — g_up≈g_down at all r (~0.11 at 0.25
  bohr, well below diamond-BC's ~0.20); no muonium contact spike. (Ignore r≈0.01
  bin, shell-volume artifact.) Matches the unrelaxed-silicon T-like result
  [[project-silicon-quantum-muon-result]].
- **Extended-radius check (2026-06-29) — RULES OUT a loose bound state too, not
  just the contact one.** Re-ran g(r) out to 6 bohr (WS inscribed radius 7.25 bohr
  for this fcc-prim supercell, Vcell=2160 bohr³, ρ_up=0.0153, ρ_dn=0.0148 e/bohr³;
  `tools/srpd_extended_radius.py silicon_bc_relaxed 6.0 120 4`, 250/1000 files =
  128k samples). Two separate things:
  (a) a SPIN-SYMMETRIC screening cloud exists — g/bulk ≈8× at contact, crosses
  bulk at ~1.5 bohr, correlation-hole dip ~0.7× at ~2 bohr, flat at bulk beyond
  ~3 bohr (~0.8 e enclosed within 1.5 bohr) — expected for a +1 charge, diamagnetic.
  (b) the NET SPIN density (the muonium discriminator) is NOT localized: cumulative
  N↑−N↓ tracks the uniform-band line (the lone doublet up-electron spread evenly
  over the cell, ∝r³, +0.42 by 6 bohr) at EVERY radius; the localized excess
  (actual − uniform) is ≈0, slightly NEGATIVE (−0.02 to −0.05) out to 6 bohr — never
  builds toward the +1 (or even +0.3) plateau a bound electron would. So the
  unpaired electron is delocalized into the bands at ALL length scales up to the
  lattice, not just inside 1 bohr. (Noise ~±0.01–0.02 ≪ the +0.3–1 a bound state
  needs → robust null.) This is the supporting baseline for EXP-003 (silicon T-relaxed
  bound-state test, `experiments/EXP-003_silicon_t_relaxed_bound_state.md`).

### Not the same site as diamond's unseeded off-T
| | Diamond unseeded (`pp_relax_2`, #8) | Silicon bc_relaxed |
|---|---|---|
| 4 nearest host /bond | **0.77**, 0.99, 1.08, 1.13 (1 host MUCH closer) | 0.96, 0.96, 1.06, 1.10 (2+2, none close) |
| site | off-T, leaning into a single carbon | near-symmetric T cage |
| dist to BC /bond | 1.51 | 1.09 |
| folded cubic-frac | ~[0.45,0.5,0.5] | ~[0,0,0.45] |
Both drifted off BC and are diamagnetic, but land at DIFFERENT interstitial
geometries. See [[project-bc-relaxed-pp-relax-2-muon-result]],
[[project-bc-seeded-muon-result]].

### Implication
Silicon's muon prefers a T cage even with the BC-relaxed lattice + default init —
exactly the unseeded failure mode that BC-seeding fixed in diamond
([[project-bc-seeded-muon-result]]: diamond bc_seeded HELD BC at 0.30 bohr). This
is the evidence motivating the **silicon bc_seeded run (#15)** — test whether
seeding pins the silicon muon at BC the way it did in diamond.

### Caveat
Silicon bc_relaxed training was ~126k steps & still descending at inference
(not converged; cf [[project-silicon-pp-energy-comparison]]). Site is robustly far
from BC and T-like, but re-run inference at plateau before quoting SRPD as final.
Reproduce: `python tools/muon_site_analysis.py silicon_bc_relaxed [spread|srpd 3.0 150]`.
