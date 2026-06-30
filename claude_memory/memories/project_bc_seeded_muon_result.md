---
name: project-bc-seeded-muon-result
description: "EXP-002 KEY RESULT: BC-seeded diamond quantum muon HOLDS the bond-centre site (0.30 bohr from BC), diamagnetic, SRPD matches classical-BC — same geometry unseeded falls to off-T trap (4.06 bohr). Proves off-T was an init trap."
metadata:
  node_type: memory
  type: project
  originSessionId: bc-seeded-analysis
---

## EXP-002 — BC-seeded diamond muon HOLDS the bond-centre site (one of our most important results)

Analysed 2026-06-27 (branch `af`). Inference job **5391609**
(`muon_d_qpp_bc_seeded_inf`) dumped all 1000 `positions_*.npy` (512k samples) into
`/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_bc_seeded/inference/positions`.
Added case **`bc_seeded`** to `tools/muon_site_analysis.py` (C_BC_RELAXED_2 geometry,
bc_pair (0,1), a=6.74, quantum muon = last of 66 particles — identical to
`bc_relaxed_pp_relax_2` except the positions dir).

### Result: the BC-seeded muon is at the BOND-CENTRE, confirmed 5 independent ways
1. **Position:** circ-mean & histogram-peak both **0.30 bohr from the relaxed BC site**
   (cubic-frac ≈[0.083,0.13,0.137] vs BC [0.125,0.125,0.125]).
2. **Coordination:** two nearest carbons C0/C1 at **2.02 / 2.21 bohr ≈ half-bond
   (2.06)**, symmetric, then a 1.3-bohr gap to the 3rd/4th (3.5 bohr). True BC.
3. **Anisotropy:** density is **oblate — tightly pinned ALONG the [111] bond
   (par RMS 0.231 bohr) but wider perpendicular (perp RMS 0.481), ratio 0.48.**
   This is the BC confinement signature (a T interstitial would be ~isotropic).
   Bond axis came out exactly [0.577,0.577,0.577]=[111].
4. **SRPD matches classical-BC almost bin-for-bin** (∫within 3 bohr: 5.91up/5.24dn
   vs classical-BC 5.97up/5.27dn). See [[project-diamond-classical-bc-muon-result]].
5. **Spin: ANISOTROPIC (bond-centred) MUONIUM — NOT diamagnetic** (CORRECTED
   2026-06-30, see extended-radius section below). The contact SRPD (rmax=1) alone
   is misleading: g_up≈g_down inside 1 bohr (that's the symmetric +1 screening
   cloud, NO 1s contact spike), so it *looks* diamagnetic — but the net spin
   (N↑−N↓ minus the uniform-band line) builds a LOCALIZED excess of ~+0.45–0.5 e⁻
   at the BOND scale (~2.5 bohr), 6–7× above uniform. The unpaired e⁻ is localized
   in the C–C bond around the muon = a loose, anisotropic muonium (bonding-orbital
   shape, not spherical 1s). (Ignore the r≈0.01 bin — shell-volume artifact.)
6. Localisation tight: RMS radius 0.51 bohr, 99% within ~1 bohr.

### The decisive contrast (EXP-002's whole point) — same geometry, only init differs
| Measure | bc_seeded (seeded@BC) | bc_relaxed_pp_relax_2 (unseeded, #8) |
|---|---|---|
| dist to BC | **0.30 bohr** | **4.06 bohr** |
| dist to C0 / C1 | 2.02 / 2.21 (BOTH ≈half-bond) | 6.14 / 2.05 (only ONE carbon close) |
| anisotropy par/perp | 0.48 (oblate, on bond) | 9.88 (sits 4 bohr off-bond by a single C) |
| site | true BC | off-T trap (next to a single carbon) |

Seeding the muon at BC makes it HOLD BC; default init falls into the off-T trap
(#8 `bc_rel_2`). **CONFIRMS the init-trap hypothesis:** the off-T result was a
sampling/init trap, not the true minimum, exactly as classical DFT predicted (BC is
the true min, ~70 mHa below off-T). See [[project-bc-relaxed-pp-relax-2-muon-result]].

### Re-run at well-trained net (2026-06-30) — CONFIRMS and tightens the result
Re-ran inference (job 5426303) on the **~176k net** (E −90.655, vs 56k/−90.42 above);
dumped all 1000 positions. Result reproduces and sharpens the 56k read:
- **Site even tighter:** circular mean cubic-frac [0.12,0.125,0.127] = **0.037 bohr
  from BC** (was 0.30 at 56k — better training pins it essentially dead-on BC); two
  carbons at 2.05/2.07 bohr ≈ half-bond, 1.6 bohr gap to rest. Hist peak 0.29 bohr.
  95.6% of samples within 1 bohr of peak.
- **Spread:** RMS radius 0.61 bohr, per-axis [0.343,0.36,0.352] (tighter/more
  isotropic at this net than the 56k oblate read; 95% within 0.98 bohr).
- **SRPD still DIAMAGNETIC:** within 1 bohr ∫ up 0.398 ≈ down 0.376; at contact
  (r≈0.08) g_up 0.196 < g_down 0.324 — no spin-up contact excess. Muonium ruled out.
VERDICT stands on the well-trained net: muon HOLDS BC, diamagnetic.

### Extended-radius net spin (2026-06-30) — the diamagnetic verdict is WRONG at BC
`tools/srpd_extended_radius.py` (now handles fixed-origin/classical cases too)
measures cumulative N↑−N↓ out to the WS radius and subtracts the uniform-band line
(`localized_excess`). Run on the 176k bc_seeded net AND on the classical fixed-BC
run (#4) as the no-ZPM comparator:

| measure | quantum bc_seeded (176k) | classical fixed-BC (#4, ckpt 222k) |
|---|---|---|
| net spin inside ~1 bohr | ≈0 (balanced screening cloud) | ≈0 |
| localized_excess PEAK | **+0.47** @ ~2.7 bohr | **+0.53** @ ~2.6 bohr |
| vs uniform line at peak | 6–7× above | 5–6× above |

Both show a LOCALIZED net spin of ~+0.5 e⁻ at the BOND scale (NOT contact, NOT
lost-to-bands) ⇒ **anisotropic, bond-centred muonium.** The classical run has the
muon CLAMPED (no ZPM) and shows the same ⇒ it's an electronic-structure property of
BC, not a delocalization/ZPM artifact. So diamond carries muonium at BOTH T (1s
contact, [[unrelaxed-pp-muon-result]]/[[diamond-t-relaxed-muon-result]]) AND BC
(bond-centred/anisotropic) — the difference is SHAPE (1s-contact vs bond-orbital),
not presence/absence. This OVERTURNS the old "BC is diamagnetic" line (which came
from looking only inside 1 bohr). Run: `python tools/srpd_extended_radius.py
bc_seeded 6.0 120 2` and `... diamond_classical_bc 4.5 90 2`. NB classical positions
are from ckpt 222k not the converged 338k → re-run that inference to tighten.
Cross-ref [[project-diamond-classical-bc-muon-result]] (its contact-only DIAMAGNETIC
verdict is likewise superseded).

### Caveats
- The 56k inference (orig analysis above) was on a NOT-converged net (E −90.42, still
  descending). The 2026-06-30 re-run at 176k (still not fully converged, but ~235 mHa
  lower) confirms site + diamagnetism, so those are no longer init/under-training
  worries. Numbers above marked "56k" are the early read; the 176k read supersedes them.
- `site_report` heuristic in `muon_site_analysis.py` is buggy: it tests the 1st→2nd
  nearest-carbon gap (mislabels a SYMMETRIC BC as "T", and a 1-close-carbon off-T as
  "BC"). Trust the explicit distances, not the printed label. (Fix: test 2nd→3rd gap.)

### How to reproduce
`python tools/muon_site_analysis.py bc_seeded` (site),
`... bc_seeded spread`, `... bc_seeded srpd 3.0 150`; bond-axis anisotropy via the
ad-hoc script (project muon disp from BC midpoint onto unit C0→C1 axis).
Training job #14 still RUNNING (5391700). [[research-goal-muon-site]]
