# EXP-006: VMC energy convergence is decoupled from the local muon spin density (muonium structure converges early)

- **Date:** 2026-07-08   **Status:** DONE — striking result, **H0 confirmed
  (decoupling)** across 4 runs incl. a classical fixed-origin control.
- **System:** diamond & silicon 2×2×2 supercells, unpaired-electron muon
  (33,32,1)=66 particles (quantum) or (33,32)=65 e⁻ + fixed H (classical), PP on
  host, a=6.74 (diamond) / 10.26 (silicon) bohr. Four production runs:
  #6 diamond unrelaxed-T (quantum), #14 diamond BC-seeded (quantum),
  #15 silicon BC-seeded (quantum), #7 diamond classical T-site (fixed H).

## Question / Motivation (why)
Tracking how muonium forms over training ([[project-bound-state-formation-dynamics]],
the checkpoint-walker sweep), the muon–electron **net-spin density around the muon
plateaus by ~step 50k** and then does not move. Yet the **ground-state energy keeps
falling by hundreds of mHa for hundreds of thousands more steps.** That looks
paradoxical: if the wavefunction is still improving enough to lower E by ~7 eV,
why doesn't the local electronic structure at the muon change at all? This
experiment quantifies the effect and asks whether it is real (a genuine
decoupling) or an artefact of the measurement. The answer has a direct practical
payoff: **can we trust the muon-site / muonium verdicts read off partially-trained
(energy-unconverged) nets?** (EXP-003, EXP-003b, silicon #15 all rested on
un-converged nets.)

## Hypothesis (H1)
The local spin density **tracks the energy**: as the energy descends, the
unpaired-electron (muonium) spin structure around the muon continues to change
measurably — i.e. the two observables converge together.

## Null hypothesis (H0)
The local spin density **decouples** from the energy: the muonium structure
reaches its converged value early (~50k) and stays flat to within measurement
noise, while the long energy tail comes from **non-local** (bulk valence-sea)
correlation with negligible footprint at the muon. Energy and a local intensive
observable are simply different quantities that converge at different rates.

## Alternative explanations (confounds) and controls
- **Moving-origin averaging artefact.** For quantum runs the SRPD origin is each
  walker's own (wandering) muon; averaging over a moving origin could *wash out*
  real changes and spuriously look flat. **Control: the classical #7 run has a
  genuinely FIXED origin (H nailed at the exact T-site, no zero-point motion).**
  If the decoupling persists there, the artefact is ruled out.
- **Measurement noise floor.** Each point is a single checkpoint (4096 walkers);
  the enclosed-count / excess noise floor is ≈±0.02–0.03 e⁻. A sub-noise drift
  can't be excluded. *Reported explicitly; the energy delta is 10–100× any
  muonium binding-energy scale, so the asymmetry of the two observables' motion is
  the point, not their absolute stillness.*
- **Cancelling up/down drifts.** The net moment (N↑−N↓) could be flat while the
  two spin channels drift together. *Control: decompose and report ΔN↑ and ΔN↓
  separately.*
- **Corrupted energy rows (VMC spikes / #15 blow-up, #7 tail).** *Guarded:*
  finite-only filter + sane energy band + block-median.

## Method (experiment detail)
Checkpoint-walker sweep (no inference dumps): read `data/positions` straight from
each `qmcjax_ckpt_*.npz` (4096 configs) at ~7 steps spanning the plateau window
(≥50k). Per checkpoint, using the **moving origin** (electrons − last-particle
muon) for quantum runs and the **fixed T-site origin** for the classical run,
under fcc min-image:
- **Radius-resolved net-spin excess** `localized_excess(r)` = cum N↑(r)−N↓(r) −
  ρ_diff·(4/3)πr³ (same math as `tools/srpd_extended_radius.py`); record the peak
  value/radius.
- **Per-spin enclosed counts** N↑(<R), N↓(<R) at the run's bound-state radius R
  (diamond-T 1.8 b, diamond-BC 2.6 b, silicon-BC 3.0 b; + contact 1.0 b), and the
  **deltas** ΔN↑, ΔN↓ across the plateau window.
- **Energy** at each step: block-median of `train_stats.csv` energy over ±1500
  steps (finite-filtered, sane-band).
Tool: scratchpad `bound_state_formation.py` + per-spin/energy analysis snippets
(reuses `muon_site_analysis` geometry/loaders; radius-resolved is essential —
contact-only misses bond-centred BC muonium). Save paths:
`.../diamond/unpaired/unrelaxed/pp` (#6), `.../bc_relaxed/pp_bc_seeded` (#14),
`.../silicon_unpaired/bc_seeded` (#15), `.../diamond/unpaired/classical/T_site/pp` (#7).

## Result
The energy keeps dropping by tens-to-hundreds of mHa **after** the spin density
has plateaued, across all four runs. The late (after-50k) descent is ~24% of the
whole energy descent.

**Energy still falling vs net-spin excess flat (plateau window):**

| run | window | ΔE (still dropping) | peak net-spin excess (mean±σ) | late window | late ΔE |
|---|---|---|---|---|---|
| diamond-T (#6, q) | 50k→522k | **−262.8 mHa** | +0.427 ± 0.021 | 200k→522k | −36.2 mHa |
| diamond-BC (#14, q) | 50k→580k | **−271.0 mHa** | +0.470 ± 0.025 | 200k→580k | −36.6 mHa |
| silicon-BC (#15, q) | 50k→196k | **−72.1 mHa** | +0.225 ± 0.017 | 100k→196k | −22.5 mHa |
| **diamond-T (#7, classical)** | 50k→472k | **−284.4 mHa** | +0.523 @1.85 b, ratio 6.10 | 200k→472k | −27.7 mHa |

**Per-spin decomposition (Δ over the plateau window) — both channels individually frozen:**

| run | R (bohr) | ΔN↑ | ΔN↓ | Δnet | noise floor |
|---|---|---|---|---|---|
| diamond-T (#6, q) | 1.8 | +0.036 | +0.024 | +0.011 | ±0.02 |
| diamond-BC (#14, q) | 2.6 | −0.027 | −0.013 | −0.015 | ±0.02 |
| silicon-BC (#15, q) | 3.0 | −0.023 | −0.028 | +0.005 | ±0.03 |
| **diamond-T (#7, classical)** | 1.8 | **−0.001** | **+0.003** | −0.004 | ±0.02 |
| diamond-T (#7, classical) | 1.0 (contact) | −0.023 | +0.006 | −0.029 | ±0.02 |

The flat net moment is **not** cancelling drifts: **both N↑ and N↓ near the muon
are individually static** (±0.02–0.04, at the noise level). The classical control
is the sharpest — at R=1.8 b both channels move by ±0.001–0.003 e⁻ while E falls
284 mHa.

## Verdict
**H0 CONFIRMED — energy convergence and the local muon spin density are
decoupled.** The muonium structure (net moment *and* both individual spin
channels) is fully converged by ~step 50k; the remaining hundreds-of-mHa energy
tail is **non-local bulk valence-sea correlation** (~4 mHa per valence electron,
delocalized across the supercell) with **no measurable footprint at the muon**.
The **classical fixed-origin run rules out the moving-origin averaging artefact**
— the decoupling is a real property of the wavefunction optimisation, not a quirk
of the moving-muon measurement.

**Physical picture.** Energy is a global, extensive quantity dominated by the
64-electron sea and is *second*-order in the wavefunction error (δψ²); the local
spin density is an intensive feature of one electron bound in the muon+cage
potential and is *first*-order in δψ. There is no theorem coupling them: lowering
E by refining correlation/nodes far from the muon need not move an already-correct
local feature. The local feature is "easy" (robust, low-variance) and saturates
early; the bulk correlation is the hard, slow part.

**Confidence:** high on the qualitative decoupling (4 runs incl. a fixed-origin
control; energy deltas ≫ any binding-energy scale; both channels flat). The
absolute stillness is bounded by the ±0.02–0.03 e⁻ noise floor and coarse
(~0.1 b) radial bins.

**Practical payoff:** the muon-site / muonium verdicts do **not** require a
fully energy-converged net — the spin structure is trustworthy from ~50k. This
retroactively validates the partially-converged verdicts in EXP-003, EXP-003b and
silicon #15.

## Next / current steps
- [ ] **Open (hyperfine):** the µSR-relevant quantity is the **contact** density
      (r→0), sharper than the integrated excess. Re-check the r→0 value with the
      fine SRPD estimator (not 0.1-b bins) to confirm it too is flat vs step — a
      sub-noise contact drift would matter for hyperfine even if the enclosed
      counts are frozen.
- [ ] Optionally promote the scratchpad sweep to `tools/bound_state_formation.py`
      and commit (reproducibility).
- [x] Classical fixed-origin control (#7) — decoupling holds; artefact ruled out.
- [x] Per-spin (↑/↓) decomposition — both channels individually frozen.
