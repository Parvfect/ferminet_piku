# EXP-003b: Does the silicon T-*seeded* quantum muon (held in the relaxed cage) form a bound state?

- **Date:** 2026-07-08   **Status:** DONE — **H0 confirmed (diamagnetic)**. This
  closes the two open follow-ups left by EXP-003.
- **System:** silicon 2×2×2 supercell (16 Si + 1 muon, doublet, PP on Si,
  a = 10.26 bohr), **T-relaxed** geometry (identical to EXP-003 / `t_relaxed.py`;
  4 coordinating Si moved ~0.0487 bohr inward, −1.1% cage contraction). The only
  difference from EXP-003 is the **MCMC seed**: the muon walkers were seeded at
  the relaxed T-site so the muon is forced to sample the relaxed cage.
  Training run **#17** (`muon_silicon_q_t_seeded`, save
  `silicon_unpaired/t_seeded`, config `configs/silicon/t_seeded.py`).

## Question / Motivation (why)
EXP-003 found the silicon T-relaxed quantum muon is **diamagnetic**, but with a
fatal caveat: the *unseeded* muon **avoided the DFT-relaxed (classical-proton)
cage entirely** (8.47 bohr away, P≈0), sitting at a roomier *unrelaxed* T-site.
So EXP-003's intended question — *does the tighter relaxed cage enable electron
binding?* — was never actually tested. EXP-003b **seeds** the muon in the relaxed
cage (à la EXP-002 bc_seeded) to ask: (a) does it **hold** there, and (b) if so,
does the tighter coordination let it **bind** an electron (muonium)?

## Hypothesis (H1)
Confined to the tighter relaxed cage, the muon–electron contact density rises
enough to form a (partial) bound state: SRPD shows a spin-up excess over
spin-down as r→0, and a localized net-spin excess that plateaus below the lattice
scale — unlike the diamagnetic unseeded runs.

## Null hypothesis (H0)
Silicon stays **diamagnetic** even with the muon held in the relaxed cage:
g_up ≈ g_down at all radii, localized net-spin excess ≈0. The diamond↔silicon
muonium contrast is a host-material property, not a cage-size / seeding artefact.

## Method (experiment detail)
Two independent read-outs, both post-hoc from walker positions (no dependence on
the live SRPD estimator):

1. **Quantum #17 (t_seeded), from checkpoints.** The paired inference
   (`inference_t_seeded.py`) never dumped `positions_*.npy` (its
   `inference/positions/` dir is empty). Instead we read the **live MCMC walkers
   straight from the training checkpoints** (`data/positions`; each ckpt holds
   `batch_size = 4096` full 66-particle configs = 33 ↑-e, 32 ↓-e, muon). Pooled
   the **last 20 converged checkpoints** (steps 196,992 → 224,000) = **81,920
   walker configs**. This is exactly the info an inference dump would give, minus
   the extra time-decorrelation (checkpoints are 2000 steps apart → effectively
   independent).
2. **Classical #11 (fixed-H t_relaxed), the contact-density upper bound.** The
   muon is a **fixed H nucleus** at the relaxed T-site (`t_relaxed_classical.py`,
   16 Si + fixed H, 65 electrons). Its 2026-07-01 inference dumped 1000 frames
   (512k samples) — verified finite (the *training* later diverged past ~330k,
   but these frames predate the blowup). SRPD uses a fixed origin = the H site.

### Files / artefacts
- **Analysis (tools, this experiment).** Added two cases to
  `tools/muon_site_analysis.CASES` and made the loaders **checkpoint-aware**:
  - `silicon_t_seeded` — quantum, reads from `checkpoints=…/t_seeded` (last
    `n_ckpt=20`) via the new `muon_site_analysis.iter_frames()` (falls back to
    checkpoints when no `positions_*.npy` exist). Geometry `SI_T_RELAXED`.
  - `silicon_classical_t_relaxed` — classical, `positions=…/classical/t_relaxed/
    inference/positions`, `n_particles=65`,
    `fixed_origin=(0.74999906,0.74999903,0.74999906)·a`.
  - `tools/srpd_extended_radius.py` refactored to consume the same
    `iter_frames()` (so it works for checkpoint-only cases too).
  Commands (ferminet-piku env):
  - `python tools/srpd_extended_radius.py silicon_t_seeded 6.0 120 1`  ← definitive test
  - `python tools/muon_site_analysis.py silicon_t_seeded spread`       ← does it hold the cage?
  - `python tools/srpd_extended_radius.py silicon_classical_t_relaxed 6.0 120 4`
  - `python tools/muon_site_analysis.py silicon_classical_t_relaxed srpd`

## Result
**Muon site (control — does the seed hold the relaxed cage?). YES.** Unlike the
unseeded #10, the t_seeded muon **holds the T-site**: histogram-peak centre =
**(7.695, 7.695, 7.695) bohr** = exactly the relaxed/design T-site
(0.75³·a); RMS radius **0.78 bohr**, per-axis (0.45, 0.45, 0.45) — tight and
**isotropic** (vs #10's anisotropic off-cage 0.81 bohr). 99% of samples within
1.49 bohr. So the binding test is now actually being performed in the relaxed
cage.

**Quantum #17 extended-radius net-spin (definitive test, out to WS 7.25 bohr,
81,920 configs).** g_up/bulk ≈ g_dn/bulk at **every** radius (contact bins
spin-symmetric). The `localized_excess` (actual N↑−N↓ minus the uniform-band
expectation) is **≈0, slightly negative** inside the lattice scale (−0.005 @1.0
bohr → −0.028 @2.7 bohr), then drifts only to **+0.08** by 5.7 bohr — it never
builds toward the +1 (or even +0.3) plateau a bound electron would show. A
spin-symmetric screening cloud is present (g/bulk ≈10× at contact) but there is
**no localized net-spin excess**.

**Classical #11 fixed-H SRPD (contact-density upper bound).** Also diamagnetic:
contact SRPD (≤1 bohr) spin-unpolarised, integral within 1 bohr **0.192 up vs
0.199 down** (down slightly higher — zero spin-up enhancement); extended-radius
`localized_excess` ≈0/negative out to ~4.5 bohr (−0.05 @3 bohr), +0.02 at large
r, no plateau. The classical upper bound confirms it is not a quantum-sampling
artefact.

## Verdict
**H0 CONFIRMED (diamagnetic).** Even with the muon **seeded and held in the
tighter DFT-relaxed T-cage** — and independently with a **classical fixed H** at
that site — silicon shows **no muonium**: no spin-up contact excess (≤1 bohr) and
no localized net-spin plateau (out to the WS radius). H1 (tighter cage → bound
state) is **rejected**. Together with EXP-003 this makes the conclusion robust:
the **diamond↔silicon muonium contrast is a host-material property**, not a
geometry / relaxation / seeding / sampling artefact. Silicon's T-site muon stays
diamagnetic whether the cage is ideal, relaxed-and-avoided, relaxed-and-held, or
occupied by a classical proton.

This closes EXP-003's two open follow-ups:
- [x] *Seed the muon in the relaxed cage* → it holds (RMS 0.78 bohr at 7.695³) and is diamagnetic.
- [x] *Classical #11 fixed-H SRPD cross-check* → diamagnetic, spin-symmetric contact.

## Caveats
- Quantum verdict read from a **partially-trained** #17 net (−62.899 @224k, still
  slowly descending) and from **checkpoint snapshots** rather than a full
  decorrelated inference dump (20 frames, 2000 steps apart). The signal is
  unambiguous and flat and matches three independent silicon baselines (#1, #9,
  #10) plus the classical #11 upper bound, so a late-step flip is very unlikely;
  the clean control would be a full inference dump at a later checkpoint.
- Classical #11 training diverged past ~330k; the analysed frames are the
  2026-07-01 inference (pre-blowup, verified finite).
