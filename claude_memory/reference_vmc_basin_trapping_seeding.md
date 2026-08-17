---
name: reference-vmc-basin-trapping-seeding
description: "Why unseeded VMC misses the BC muon site (basin-trapping / mode collapse), why seeding is NOT cheating, and the µSR hyperfine caveats — the conceptual framing behind the seeded-site methodology"
metadata: 
  node_type: memory
  type: reference
  originSessionId: ce63bd55-f2e8-4ebf-9577-06954c0ecdad
---

The physics/optimisation reasoning that explains the whole muon-site investigation
(EXP-001/002/004/005). Use this to interpret and defend the seeded-site methodology.

**Why unseeded VMC misses BC (the mechanism).** The muon site is a *basin-selection /
reachability* problem, NOT a stability problem. BC is a genuine, deep minimum (BC-seeded
GSE < T < off-T — confirmed), but from a symmetric/diffuse init the optimiser flows into
a T-cage instead. Three features make this problem hard where ordinary electronic ground
states are easy: (1) **near-degenerate multi-well** landscape — BC-vs-T is meV–tens-of-meV
on top of ~90 Ha, so the distinguishing gradient is buried in noise; (2) **heavy
distinguishable particle** (muon 207×e) — localised eigenstates, tunnelling exponentially
suppressed, so ψ *can* localise in one well AND MCMC physically can't hop the muon between
cages (why frozen-wide EXP-005 didn't help — cross-cell acceptance ≈ 0); (3) **BC's
stabilisation is muonium** (muon binds an e⁻), a *correlation feature the fresh net doesn't
represent*, so putting amplitude at BC yields NO early gradient reward while a T-cage is
"electrostatically easy" → gradient at step 0 points away from BC (chicken-and-egg). Then
**mode collapse locks it in**: once amplitude commits to a cage, 0 samples reach BC → 0
gradient signal there → ψ→0 outside the cage → the deeper well is invisible. VMC's only
guarantee is variational-bound + convergence to a *local* min; it finds the global GS only
when the landscape is benign (usual electronic case), which this isn't. Same class as
polaron self-trapping / Anderson localisation / defect levels — the known-hard cases.

**Why seeding works & is NOT cheating.** Seeding sets the MCMC *initial condition* only —
it does NOT touch the Hamiltonian, add a constraint/bias potential, or change what
Metropolis samples (still true |ψ|²). It changes the optimisation *path*, not the
*objective*; the variational bound E[ψ]≥E₀ holds for any ψ regardless of how obtained, so
a seeded-converged energy is a rigorous upper bound. (Contrast: a clamped/classical muon
DOES change the physics — no ZPM.) After seeding the muon is fully quantum and free to
leave — that it *holds* BC is a RESULT, not an imposition. Standard practice: broken-symmetry
UHF/DFT, ΔSCF, informed initial orbitals/geometry. Seeding breaks the chicken-and-egg
(supplies amplitude → gradient can build muonium → deepen) AND only holds because BC is a
true min. "Never sees samples outside BC" is the lock-in that *maintains* it, but that's
SYMMETRIC (seeding T also locks in) — the asymmetry that makes BC correct is the lower
energy. **Seed-dependence of the converged energy is itself the diagnostic of a
basin-trapped landscape; the lowest-energy seed = best GS estimate** (basin-hopping by hand).

**µSR hyperfine caveats (the honest limits).** For the stated goal — muon hyperfine in
semiconductors — seeding is arguably *more* correct than unseeded, because real µSR sees
*kinetically-trapped metastable* sites (e.g. Si: Mu_BC and Mu_T coexist, T/doping-dependent),
so you WANT per-site hyperfine and a global-min-seeking optimiser would give the wrong one.
Discipline: (1) seeding only finds sites you seed — can't discover an unguessed one (⇒
envelope-bias / unbiased run as independent check); (2) confirm each seeded site HOLDS and
is a true local min before quoting hyperfine; (3) hyperfine is a *local, cusp-dominated*
(contact-density) quantity — ansatz bias in the e⁻–muon cusp limits accuracy even at the
right site/energy (separate from the site question, but what actually bounds the numbers);
(4) if wells tunnel-couple the observed hyperfine is a *motional average* over wells that
single-site seeding misses (small for a heavy muon but known for Mu_BC in Si).
