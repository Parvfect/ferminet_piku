# EXP-001: Is the FermiNet off-T muon the true site, or a sampling trap?

- **Date:** 2026-06-24   **Status:** done (H-only relaxations confirming)
- **System:** diamond 2×2 supercell, BC-relaxed cage (`pp_relax_2`, open-shell
  DFT geometry), 16 C + 1 muon, doublet (net spin 1).

## Question / Motivation (why)
The FermiNet inference for `pp_relax_2` localised the quantum muon at an
asymmetric **off-T** interstitial site (~0.32 bohr off the ideal T-site, one C at
~2.0 bohr), **not** at the bond-centre (BC) we had explicitly DFT-relaxed it
toward (4.06 bohr away). Puzzlingly, this run's VMC energy sat ~6 mHa *below* the
converged unrelaxed-T run. We needed to know: **is off-T genuinely the
lowest-energy muon site, or is the quantum muon trapped there by the way walkers
are sampled?**

This matters because the whole research goal is to pin the muon site. If the
sampling traps the muon, *every* relaxed-geometry FermiNet run could be reporting
the wrong site, and the energetics comparisons built on them are invalid.

We use **classical DFT (QE, PBE)** as an independent check: it evaluates the
Born–Oppenheimer energy at *any fixed* muon position, so it has **no MCMC
ergodicity problem** — it can directly compare BC vs off-T, which the single
FermiNet chain cannot. (The muon is modelled as a clamped H/proton; correct for
the classical potential-energy surface, since charge — not mass — sets the PES.)

## Hypothesis (H1)
The off-T localisation is a **sampling/initialisation artifact**. BC is the true
(global) minimum; off-T is a higher local minimum. The carbon-centred walker init
(`init.py`: muon at a random C + 1 bohr Gaussian) plus local-move Metropolis plus
VMC self-consistency trap the muon in the off-T basin, which is separated from BC
by the **C1 nucleus** (BC, C1, off-T and the ideal T-site are collinear on the
[111] diagonal, with C1 lying *between* BC and off-T).

## Null hypothesis (H0)
Off-T is the true classical minimum: **E(off-T) ≤ E(BC)** in the frozen
BC-relaxed cage. If true, the FermiNet result is physically correct and there is
no trap; the energy gradient near off-T points inward from both sides of C1.

## Alternative explanations (and controls)
- *Convergence artifact in the VMC energy* → controlled: we don't rely on the VMC
  energy here at all; classical DFT is self-contained and converged
  (`conv_thr=1e-8`).
- *Classical ≠ quantum (ZPE could flip the ordering)* → addressed in the verdict:
  the BC–offT gap is compared against a realistic muon ZPE scale (~0.1–0.3 eV).
- *off-T position mis-identified* → controlled: we use the FermiNet histogram-peak
  muon centre, and confirm the force there is ≈ 0 (it is a stationary point).
- *Geometry/Hamiltonian mismatch* → controlled: identical cell, k-points, ecut,
  PPs, spin as the relaxation that *defined* the cage; only the muon position
  varies between calculations.

## Method (experiment detail)
QE 7.4.1 `pw.x`, `calculation='scf'`, `tprnfor=.true.`, PBE PAW, `ecutwfc=50`,
`ecutrho=400`, `2×2×2` k-points, `nspin=2`, `tot_magnetization=1`, `nosym`.
Carbons **frozen** at the BC-relaxed positions (from
`bc_2_relax_unpaired.out` final coords). The muon (H) is placed, one calc each, at
four points on the [111] diagonal:

| label | cubic-frac | what |
|-------|-----------|------|
| `bc` | 0.125 | the DFT-relaxed bond centre |
| `offT` | 0.474 | FermiNet quantum-muon site |
| `Bminus_toBC` | 0.216 | 1 bohr from C1, **BC-side** |
| `Aplus_toT` | 0.387 | 1 bohr from C1, **T-side** |

Inputs/outputs: `~/qe/diamond_test/force_{bc,offT,Aplus_toT,Bminus_toBC}.{in,out}`.
Input generator: `scratchpad/gen_force_inputs.py`. Ran on the login node
(`pw.x`, OMP=32, ~1 min each). Follow-up: H-only relaxations (carbons frozen,
`if_pos` = 0 0 0 for C, 1 1 1 for H) from each side of C1 —
`relax_from_{Aplus_toT,Bminus_toBC}.{in,out}` — to confirm the basins.

## Result
Same frozen cage, only the muon moved:

| muon position | cubic-frac | force on muon (Ry/au) | direction | energy (Ry) | ΔE vs BC |
|---|---|---|---|---|---|
| **BC** | 0.125 | ≈ 0 (−1e-6) | stationary | **−295.60317** | 0 |
| BC-side of C1 | 0.216 | −2.97·[111] | → toward **BC** | −294.16415 | +1.44 eV |
| T-side of C1 | 0.387 | +2.95·[111] | → toward **off-T** | −294.06443 | +1.54 eV |
| **off-T** | 0.474 | ≈ 0 (~0.01/comp) | local min | **−295.46363** | **+1.90 eV** |

- BC energy reproduces the relaxation's final energy exactly (−295.60317 Ry) → consistency check passes.
- **ΔE(off-T − BC) = 0.1395 Ry = 1.90 eV = 70 mHa.**
- The force flips sign across C1: a muon on the T-side rolls to off-T; one on the
  BC-side rolls to BC. The basin **watershed is the C1 nucleus** — the two minima
  are not connected without passing through the carbon.

## Verdict
**H0 rejected.** BC is the true minimum, **1.90 eV (70 mHa) below** the off-T site;
off-T is a higher local minimum. The FermiNet quantum muon is **trapped** at off-T:
the carbon-centred init lands it on the T-side of C1, local-move MCMC cannot climb
past the C1 nucleus, and VMC self-consistency reinforces the occupied basin.

1.90 eV is an order of magnitude larger than any plausible muon zero-point-energy
difference (~0.1–0.3 eV; the more open off-T site only gains a little), so **BC is
the muon's true site quantum-mechanically as well.** This overturns the earlier
"off-T is genuinely preferred" reading. By implication, the other diamond
relaxed-geometry FermiNet runs that report T/off-T are likely trapped the same way.

## Unresolved — why does the optimiser SELECT off-T over the lower BC?
Rejecting H0 told us BC is the global minimum; it did **not** tell us why the
FermiNet quantum muon picks off-T. Follow-up checks (2026-06-24) deepen the puzzle
rather than resolve it:

- **Total trap, confirmed.** Across 512k inference samples, **0.000%** come within
  3.24 bohr of BC; 100% sit within 1.5 bohr of off-T. The trained wavefunction has
  zero amplitude at BC — no minor secondary peak.
- **The initialisation is NOT biased against BC.** Replicating `init.py` (random C
  + 1 bohr Gaussian, 200k draws): ~1.70% of walkers start within 1.5 bohr of BC vs
  ~1.66% within 1.5 bohr of off-T — essentially **equal** support in both basins
  (44.7% start closer to BC; 72% of C0/C1-seeded walkers lean BC). The earlier
  "init statistics favour the T-family" idea is **refuted**.

So forces (symmetric, ±2.95 Ry/au across C1), init (symmetric), and energetics
(favour BC) all fail to explain the 100% → off-T collapse. The selection must be a
property of the variational **optimisation dynamics**: a heavy muon localises to a
single peak; with equal support in two non-degenerate basins the symmetric state is
unstable and breaks to one peak; once the peak sits at off-T, migrating it to the
lower BC basin means pushing density through/around the C1 barrier (higher-energy
intermediate wavefunctions), which gradient descent will not do. off-T is thus a
metastable fixed point of the optimiser.

**Why off-T specifically wins is undetermined.** Two candidates, not yet distinguished:
- **(a) spontaneous symmetry breaking** — seed-dependent; another RNG seed might give BC.
- **(b) network representability bias** — an open interstitial site may be intrinsically
  easier to localise early than an in-bond site, favouring off-T at every seed.

Tests that would resolve it:
- **Multiple random seeds, same `pp_relax_2` setup** — always off-T ⇒ (b); sometimes BC ⇒ (a).
- **BC-seeded run** (walkers initialised at BC) — does the net *hold* the deeper BC
  basin (off-T was a trap) or drain to off-T regardless?

## Next / current steps
- **[done]** H-only DFT relaxations confirm the watershed: T-side start → off-T
  (frac 0.473, −295.46387 Ry); BC-side start → BC (frac 0.125, −295.60317 Ry).
  Endpoints/energies match the single-points. Files: `relax_from_*.out`.
- **BC-seeded quantum FermiNet run** — no longer needed to decide the *site*
  (classical settles it), but still wanted for the true *quantum* BC energy + ZPE
  feeding the SRPD/hyperfine story. Requires the `muon_init_coord` change in
  `init.py`/`base_config.py` and a fresh-net seed at BC (warm-start drains back to
  off-T). See [[project-bc-relaxed-pp-relax-2-muon-result]].
- **Audit other diamond runs** (unrelaxed, `pp`, `pp_new`, `nopp`) for the same
  off-T trap vs their relaxed BC.
