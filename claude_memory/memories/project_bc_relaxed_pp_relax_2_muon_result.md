---
name: project-bc-relaxed-pp-relax-2-muon-result
description: "bc_relaxed pp_relax_2 (2nd open-shell DFT-relaxed BC geom) inference: muon migrates to T-site (0.32 bohr from ideal T, 4.06 from BC); DIAMAGNETIC (no muonium)"
metadata: 
  node_type: memory
  type: project
  originSessionId: be058652-264b-4de2-b436-0f12122b7ad4
---

Inference (job 5360284, `muon_d_qpp_bc_rel_2_inf`) on the 2nd BC-relaxed PP
diamond run (`pp_relax_2`, open-shell/unpaired-electron DFT geometry, training
job 5344740 step 280196 E=−90.65). 1000 positions files / 512k muon samples.
Analysis case `bc_relaxed_pp_relax_2` added to tools/muon_site_analysis.py
(carbons = `C_BC_RELAXED_2`, bc_pair (0,1), a=6.74).

**Site = asymmetric OFF-T site, NOT the seeded BC site, NOT the symmetric T.**
Muon centre ~0.32 bohr from ideal T (0.5,0.5,0.5), 4.06 bohr from relaxed BC.
Tight (R≈0.97), RMS radius 0.54 bohr, isotropic, 95% within 0.86 bohr. BUT the
nearest-C profile is asymmetric: ONE carbon (C1) at ~2.0 bohr (≈ C–Mu bond length
2.06) + rest at 2.7–3.1 — NOT the symmetric T cage. Contrast unrelaxed muon: 4 C
all at 2.92 bohr (clean symmetric T). So this is a lower-symmetry, more-bound
C–Mu-bonded site enabled by the BC bond-stretch pushing C1 (frac 0.25→0.30) into
the interstitial. (Earlier note called it "T-site" — refine to "off-T bonded".)

**ENERGY PUZZLE RESOLVED.** pp_relax_2 (this run, #8) E=−90.66904±0.00029 is
~6 mHa BELOW converged unrelaxed (#6) −90.66312±0.00017, and STILL DROPPING
(NOT converged, −1.5e-7/step at step 280k). Why a strained cage beats the ideal
cage: the two runs are DIFFERENT defect states, not "same T-muon, two lattices."
Carbons are CLAMPED, only the muon is quantum — relaxing the cage for BC does NOT
pin the muon at BC; it just deforms the cage, and the muon drops into the off-T
bonded site (one C at ~2.0 bohr). Extra C–Mu binding outweighs the lattice strain
→ lower E. Means: symmetric ideal-T is NOT the global min; a small distortion
gives a lower site. Does NOT give true relaxed muon energy/geometry (cage never
reoptimised for where muon went) and is NOT a BC-vs-T comparison (muon not at BC).
Clean comparison needs the T-relaxed run [[project-t-relaxed-runs]] (relax cage
WITH muon at T) + classical muon constrained at BC.

**CLASSICAL DFT VERDICT (2026-06-24, OVERTURNS the above "off-T preferred" lean):**
Ran QE single-point forces (PBE, nspin2 doublet, frozen BC-relaxed carbons, H =
classical muon) at 4 muon positions on the [111] diagonal (files in
~/qe/diamond_test/force_*.{in,out}; ran on LOGIN NODE via pw.x OMP=32, ~1 min
each — slurm queue too slow). Energies (Ry), same frozen cage, only H moved:
  BC (frac 0.125)   −295.60317  force≈0 (deep global min; == relax energy)
  off-T (0.474)     −295.46363  force≈0 (LOCAL min, +0.1395 Ry = +1.90 eV = +70 mHa)
  1b from C1 BC-side(0.216) force −2.97·[111] → rolls to BC
  1b from C1 T-side (0.387) force +2.95·[111] → rolls to off-T
KEY: **BC is the true minimum, ~1.90 eV (70 mHa) BELOW off-T.** off-T is a higher
LOCAL min. The watershed is C1: BC & off-T sit on opposite sides of the C1 nucleus
on the [111] diagonal, so a muon can't cross between them without passing through
C1. FermiNet's carbon-centered init + local MCMC + self-consistency TRAP the
quantum muon in the off-T basin; it never reaches the much-lower BC. 1.90 eV is far
beyond any ZPE difference (~0.1–0.3 eV, and off-T's openness only helps it by that
little), so BC is the muon's true site quantum-mechanically too. Earlier
"equal-reachability ⇒ off-T genuinely lower" reasoning was WRONG — off-T is a trap.
Likely ALL diamond relaxed-geom quantum runs are similarly trapped at T/off-T.
Confirms the user's init-trap hunch. Next: BC-seeded quantum FermiNet run to get
the true quantum BC energy (still wanted, but classical already settles the site).

**REFINEMENT (selection mechanism still OPEN):** Checked the FermiNet samples —
0.000% within 3.24 bohr of BC, 100% at off-T (total trap, no secondary peak).
Replicated init.py (random C + 1 bohr Gaussian): ~1.70% of walkers start within
1.5 bohr of BC vs ~1.66% within 1.5 bohr of off-T — init is SYMMETRIC, NOT
T-biased. So "init favours T-family" is REFUTED. Forces (symmetric), init
(symmetric), energetics (favour BC) all fail to explain the 100%→off-T collapse;
it's a variational-optimisation symmetry-breaking trap, and WHY off-T specifically
wins (random seed-dependent SSB vs network representability bias) is undetermined.

**EXPERIMENTS DIR (new):** `ferminet_piku/experiments/` — lab notebook, one file
per experiment (template + index in README.md). EXP-001 = the classical
forces/extremals study (DONE, H0 rejected) incl. an "Unresolved" section on the
selection question. EXP-002 = BC-seeded quantum run, **IMPLEMENTED & LAUNCHED
2026-06-24** (diamond job 5370014 `muon_d_qpp_bc_seeded`, PENDING). Code done +
committed: init.py `init_electrons` gains `muon_init_coord`/`muon_init_width`
overriding the last/muon particle; init_mcmc_data passthrough; base_config.py mcmc
defaults (None / 0.5); new `configs/diamond/bc_relaxed/bc_seeded.py` (muon seeded
at 0.125*a width 0.3, fresh net restore_path==save_path empty, load_data=True).
Silicon analogue `silicon/bc_seeded.py` (#15) STAGED but NOT started — launch only
after diamond #14 verified. Sanity check still owed once #14 RUNS: confirm
"Training new model" log + muon at BC (≈0.125 frac) + no early NaN. NB: init width
only sets the START cloud, it does NOT confine the muon (MCMC+ψ govern later
motion). QE force/relax files in ~/qe/diamond_test/force_*.{in,out},
relax_from_*.out (login-node runs).

**SRPD = DIAMAGNETIC (no muonium).** g_up ≈ g_down at all radii; integral within
1.0 bohr 0.351 up vs 0.350 down. Fine-bin r=0.003 g_down=3.73 spike is the known
1-count artifact (vanishes at nbins=10: first shell g_up 0.287 vs g_down 0.239).

**Consistency:** matches all-electron [[project-bc-relaxed-nopp-muon-result]]
(diamagnetic, T-site) and confirms ALL diamond runs localise at T (see
[[project-unrelaxed-pp-muon-result]], [[project-bc-relaxed-pp-muon-result]],
[[project-bc-relaxed-pp-new-muon]]). Contrast: pp / pp_new PP runs showed WEAK
muonium at T; this pp_relax_2 run is diamagnetic like nopp. Supports the standing
[[research-goal-muon-site]]: relaxed BC is NOT stable — muon → T-site.
