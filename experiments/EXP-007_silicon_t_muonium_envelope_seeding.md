# EXP-007: Can we bind silicon T-site muonium? (muon-anchored envelope + electron seeding + finite size)

- **Date:** 2026-07-08   **Status:** PLANNED (design + rationale; not yet run)
- **System:** silicon 2×2×2 supercell, unpaired-electron muon at the tetrahedral
  interstitial (33,32,1)=66, PP on Si, a=10.26 bohr — and a larger 3×3×3
  (54-atom) variant for the finite-size arm.

## Question / Motivation (why)
Every silicon **T-site** run we have is **DIAMAGNETIC** — unrelaxed (#1),
t_relaxed (#10), t_seeded (#17, holds the cage), and even the classical fixed-H
(#11) — the unpaired electron delocalizes into the bands instead of binding to the
muon. **But experimentally the silicon T-site is paramagnetic:** µSR sees *normal
muonium* Mu⁰_T (isotropic, hyperfine ≈45% of vacuum) at the T-site, alongside
anomalous Mu\* at BC. We reproduce (weak) BC muonium (#15, +0.25) but miss Mu_T
entirely. **So the diamagnetic silicon-T result is almost certainly a
methodological artefact, not physics.** This experiment asks what it takes to bind
it — which both (a) would validate the muon-FermiNet method against known µSR data
and (b) isolates *why* silicon differs from diamond.

What we have already RULED OUT as the cause (so the lever is on the *electron*
side, not the muon):
- **Not muon site / seeding:** t_seeded holds the relaxed T-cage yet stays
  diamagnetic ([[EXP-003b]]).
- **Not muon zero-point motion:** the classical fixed-H T-muon is also diamagnetic
  (silicon #11), while classical diamond-T binds strongly (+0.523, EXP-006).

## Leading mechanism — two coupled causes on the electron side

**(1) The ansatz cannot cheaply *represent* a muon-localized orbital.** In PBC runs
the multiplicative envelope is `pbc.envelopes.make_multiwave_envelope`
(`train.py:459-464`): a **plane-wave / Bloch form**
`Σ_k σ_k[cos/sin](k·r_ae)` centred on each **nucleus**. It has **no decaying,
localized component**, and — because the quantum muon is a *walker particle*, not
an atom in `charges` — **the muon is never an envelope centre**. So a
muon-bound (localized, decaying) orbital has to be built by the MLP fighting a
plane-wave envelope with no anchoring basis function at the muon. In diamond's
tight C-cage the surrounding nuclei's structure reaches the T-site and the network
manages it; in silicon's roomy cage it does not.

**(2) Silicon muonium is spatially large (finite-size).** High ε (≈11.7 vs
diamond 5.7) and light CB mass expand the bound electron. Evidence from our own
data: the one silicon state that *did* bind (#15 BC) peaks at **~3 bohr and is
still rising at 3.4 bohr**, already straining the 2×2×2 cell (WS radius 7.25 b),
whereas diamond's compact muonium plateaus by ~2.6 b. A diffuse T-muonium electron
in a 16-atom cell at Γ overlaps its periodic images and hybridizes into the
conduction-band manifold → reads diamagnetic.

**The 2×2 factorial we already have** (host × muon-as-envelope-centre): classical
runs make the muon an `'H'` atom → it *gets* an envelope centre; quantum runs do
not. Diamond binds in **both**; silicon binds in **neither**. → an envelope centre
alone is *not sufficient* for silicon. **But** the classical H envelope is a
*tight* hydrogenic `exp(-σr)` with σ~1 b⁻¹ (small radius), the wrong shape for
silicon's *diffuse* muonium. So the right test is a **diffuse** muon envelope
(large radius), not a hydrogenic one — and likely in a **larger cell**.

## Hypothesis (H1)
Giving the ansatz a **diffuse, muon-anchored decaying envelope** (a localized
basis function that can host an expanded bound electron), **seeding an up-electron
on the muon** (so the optimizer finds the muonium basin), and/or **enlarging the
cell** (so the bound electron fits) produces a **localized net-spin excess at the
T-site** — a paramagnetic Mu_T with contact g↑>g↓ — reproducing the experimental
normal muonium.

## Null hypothesis (H0)
Silicon T stays diamagnetic under all three interventions → a deeper gap (the
plane-wave-envelope ansatz cannot represent this state even when anchored/seeded,
or the 3×3×3 cell is still too small), and the diamond↔silicon contrast is not
merely representability/finite-size.

## Alternative explanations (confounds) & controls
- **Representability vs finite-size** are entangled. *Control — staged design:*
  run the cheap representability test (envelope+seed) in the *current* cell FIRST;
  if a bound state forms and holds → representability. If it forms then drains →
  finite-size; escalate to 3×3×3. The classical-Si datum already argues finite
  size is a real co-factor.
- **Seeding a basin vs the true GS.** A seed/envelope could force a *spurious*
  bound state. *Control:* the bound state must be **energetically favourable**
  (lower or equal VMC energy vs the diamagnetic run) and **stable after the seed
  is released** and after σ/π are free to unwind — not just an imprinted artefact.
  Cross-check the hyperfine against the experimental ≈45%-of-vacuum value.
- **Envelope σ mistuned.** Too tight → can't host the diffuse electron (the
  classical-H failure mode); too loose → indistinguishable from a band state.
  *Control:* make σ **learnable** (per-orbital), init to a diffuse value
  (σ ≈ 0.25–0.5 b⁻¹, radius ~2–4 b) and let training set it.

## Method — the envelope suggestion (concrete, code-grounded)

**Add a diffuse, muon-anchored isotropic decaying envelope, multiplied into the
existing plane-wave PBC envelope** (PRE_DETERMINANT), so bulk bands keep the Bloch
envelope while a muon-localized orbital becomes representable:

```
env_i(r_e) = [ multiwave_env_i(r_ae over Si nuclei) ]           # unchanged, bulk bands
           * [ 1 + Σ_i π^μ_i · exp(-σ^μ_i · |r_e − r_μ|) ]      # NEW: muon-anchored, decaying
```

- `r_μ` = the **live muon walker position** (last particle, index 65), NOT a fixed
  nucleus — so the term must be fed the electron–muon displacement, unlike the
  atom envelopes which use fixed `ae`/`r_ae`.
- `σ^μ_i`, `π^μ_i` **learnable** per orbital/output-dim; init `π^μ≈0` (so the term
  starts as a small +perturbation and doesn't disturb the converged bulk) with a
  **diffuse** `σ^μ ≈ 0.3 b⁻¹` (radius ~3 b, matching silicon's expanded muonium —
  deliberately looser than the hydrogenic classical-H envelope that failed).

**Implementation touch-points:**
- `ferminet/pbc/envelopes.py`: new constructor
  `make_multiwave_plus_muon_envelope(kpoints, muon_idx)` reusing
  `make_multiwave_envelope` for the Si part and `envelopes.make_isotropic_envelope`
  form `exp(-σ r)·π` for the muon part.
- `ferminet/networks.py` (envelope application, ~L1460-1470): compute
  `r_eμ = |electron_pos − muon_pos|` from the particle positions (muon = last
  particle) and pass it to the new envelope term. The atom envelopes already get
  `ae`/`r_ae`; this adds one moving centre.
- `train.py:459-464`: select the new envelope when a config flag
  (e.g. `cfg.network.muon_envelope = True`) is set, so stock runs are unchanged.

**Complementary levers (compose with the envelope):**
1. **Electron seeding** (`init.py`, mirroring the existing `mcmc.muon_init_coord`
   machinery): initialize one up-spin walker coordinate near the muon so early
   gradients feel the muonium configuration (basin selection; identity-free — the
   seeded slot washes out under antisymmetry). Add `mcmc.electron_init_coord`
   /`_width` paralleling the muon knobs. **Written up as its own experiment,
   [[EXP-008]]** — the cheapest lever (no ansatz change) and a clean
   basin-vs-representability discriminator; run it as stage-1 here.
2. **Electron–muon Jastrow cusp** (`jastrows.py`): an attractive two-body factor
   with the correct +1 cusp on e–μ pairs — cheaper to wire than a moving envelope
   centre (the Jastrow already sees all particle pairs incl. the muon), and pulls
   density onto the muon.
3. **Finite-size arm:** a 3×3×3 (54-atom) silicon T-site config + SLURM launcher
   (new `configs/silicon/t_seeded_3x3x3.py` + `jobs/silicon/*.sh`), Γ-point, to
   test whether the bound electron just needs room. Mind that Si's CBM is not at Γ
   — check the muonium level sits in the gap, not resonant with a folded band.

**Staged run plan (cheap → expensive):**
1. current cell + muon envelope + electron seed (+ optional e–μ Jastrow). Read
   `localized_excess(r)` and contact g↑/g↓ vs step from checkpoints
   ([[project-bound-state-formation-dynamics]] tooling — no inference dump needed).
2. If (1) forms-then-drains → 3×3×3 cell, same ansatz.

## Result
_(pending — experiment not yet run)_

**Success criteria:** a localized net-spin excess at the T-site that **plateaus**
(not tracks the uniform band), contact **g↑/g↓ > 1**, `localized_excess` rising to
≳+0.3 and stable after seed release; ideally a contact density consistent with the
experimental Mu_T hyperfine (~45% of vacuum muonium). Energy must be ≤ the
diamagnetic run.

## Verdict
_(pending)_

## Next / current steps
- [ ] Implement `make_multiwave_plus_muon_envelope` + networks plumbing +
      `cfg.network.muon_envelope` flag (stock runs unchanged).
- [ ] Add `mcmc.electron_init_coord`/`_width` seeding in `init.py`.
- [ ] Stage-1 run: current cell, envelope + electron seed; checkpoint-sweep the
      contact SRPD vs step.
- [ ] If forms-then-drains: build the 3×3×3 silicon-T config + launcher (finite-size arm).
- [ ] (Optional) e–μ Jastrow cusp as an independent/compounding lever.
- [ ] Cross-check any bound state's hyperfine vs the experimental Mu_T value.

## Relation to prior experiments
Builds on [[EXP-003]]/[[EXP-003b]] (silicon T diamagnetic, muon-site ruled out),
[[EXP-006]] (energy⟂local-spin decoupling → verdicts readable from ~50k, so these
tests need only partial training), and the EXP-004/005 lead that the next lever
for muon localization is **envelope-level** bias, not sampling. The
[[reference-vmc-basin-trapping-seeding]] note (seeding = init, not cheating)
covers why electron seeding is legitimate.
