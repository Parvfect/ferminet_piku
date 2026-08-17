# Muon-site FermiNet — Project Context & Results

_A quantum Monte Carlo study of muon localisation and muonium formation in
diamond and silicon. Last updated 2026-08-17._

---

## 1. Introduction

When a positive muon (µ⁺) is implanted in a semiconductor it stops at an
interstitial site and may capture an electron to form **muonium** (Mu = µ⁺ + e⁻),
a light hydrogen isotope analogue. Muon spin rotation (**µSR**) experiments read
out the muon's local electronic environment via the hyperfine coupling, so the
two questions that decide the interpretation of any µSR spectrum are:

1. **Where does the muon sit?** — the **tetrahedral interstitial (T-site)** or the
   **bond-centre (BC)** between two host atoms.
2. **Is it paramagnetic or diamagnetic?** — does it bind an unpaired electron
   (**muonium**, paramagnetic) or does the electron delocalise into the host bands
   (**diamagnetic**, a screened µ⁺)?

This project attacks both questions from first principles using **FermiNet**, a
neural-network variational Monte Carlo (VMC) ansatz for many-electron
wavefunctions, extended so that the muon is treated as an explicit **quantum
particle** with its own zero-point motion (ZPM) — not a fixed classical point
charge. This lets us test, without empirical input, whether a site is a true
energy minimum and whether an electron actually binds there.

The scientific payoff is twofold: (a) a parameter-free prediction of muon
site + muonium character in two archetypal covalent semiconductors, and (b) a
methodological demonstration that many-body ML-VMC can resolve a subtle,
spatially-local electronic-structure question (a bound spin ½ electron) inside a
periodic bulk.

**One-sentence result:** *diamond forms muonium at both the T and BC sites, while
silicon's T-site is diamagnetic in every simulation we can construct — a
host-material contrast that is robust to lattice relaxation, walker seeding, and
the classical-vs-quantum treatment of the muon.*

---

## 2. Method

**Fork of DeepMind FermiNet (JAX VMC).** The upstream ansatz models electrons
only; this fork generalises to arbitrary particle species via three parallel
tuples in the config: `particles` (counts per species), `charges`, and `masses`
(in electron masses; muon = 206.768 mₑ). The muon is the **last particle**.

- **Quantum muon** = doublet `(33, 32, 1)` — 33 ↑-electrons, 32 ↓-electrons, 1
  muon (66 particles total, net charge +1 with a spin-½ electron surplus).
- **Classical muon** = a fixed `'H'` nucleus at the chosen site with `(33, 32)`
  electrons — an upper bound on contact density and a fixed-origin control (no
  ZPM).

**Solid-state setup.** 2×2×2 supercells of diamond (a = 6.74 bohr) and silicon
(a = 10.26 bohr), periodic boundary conditions (`ferminet/pbc/`), pseudopotentials
on the host atoms, pretraining disabled. Geometries are DFT-relaxed lattice
coordinates (Quantum ESPRESSO), including charged-state relaxations for the BC
cage.

**Seeding (a key experimental lever).** The muon's MCMC walkers can be initialised
near a chosen site (`cfg.mcmc.muon_init_coord/width`). This tests whether a site is
a **true minimum** vs a **sampling trap**: if walkers seeded at a site *stay*
there and the energy is competitive, the site is real; if they drain away, it is
not. Seeding sets only the initial walker positions, not the stationary |ψ|², so
it is initialisation, not bias.

**Analysis (runs on CPU, no cluster needed).**
- `tools/energy_convergence.py` — block-averaged VMC energy + drift test from
  `train_stats.csv`.
- `tools/muon_site_analysis.py` — muon site, spread, contact spin density; FCC
  min-image geometry.
- `tools/srpd_extended_radius.py` — the muonium discriminator: the
  **radius-resolved net-spin excess** `localized_excess(r)` = cumulative
  N↑(r) − N↓(r) minus the uniform-band baseline. Plateau to +0.3…+1.0 within
  lattice scale ⇒ a bound electron (**muonium**); ≈ 0 at all radii ⇒
  **diamagnetic**. Both tools are **checkpoint-aware** (`iter_frames`): with no
  inference dump they read the 4096 live walkers from each checkpoint's
  `data/positions`, pooling the last ~20 checkpoints for statistics.

> **Critical measurement subtlety:** a *contact-only* SRPD (r ≤ 1 bohr) sees only
> the screening cloud and mislabels **bond-centred** muonium as "diamagnetic."
> The extended-radius net-spin excess is essential — it was the historical source
> of a since-corrected "BC is diamagnetic" error.

---

## 3. Results

### 3.1 Diamond — muonium at BOTH sites

Diamond binds an unpaired electron to the muon at both interstitial sites; the
**difference is the shape of the bound state, not its presence.**

- **T-site (contact / atomic muonium).** The unrelaxed quantum T-muon (#6) forms
  muonium with net-spin excess **+0.42**, a strong contact enhancement
  (g↑/g↓ ≈ 4.5–6.9), and a cloud that **contracts** over training (diffuse →
  atomic 1s-like). The classical fixed-T control (#7, converged
  E = −90.696 Ha) confirms it independently: **+0.523** at 1.85 bohr, contact
  ratio 6.1 — with no ZPM, so the binding is electronic, not a motion artefact.
- **BC-site (bond-orbital muonium).** The BC-seeded quantum muon (#14) **holds the
  bond centre** (0.037 bohr from BC at convergence; two carbons at ~half-bond) and
  carries an **anisotropic, bond-centred** muonium: net-spin excess **+0.47** as a
  *shell* at ~2.6 bohr with **zero contact enhancement** (g↑/g↓ ≈ 1). This is the
  most important run — it is also the **lowest-energy quantum diamond result**
  (≈ −90.697 Ha), below the off-T trap (−90.669), matching the classical PES.
- **Relaxed T-cage (#12).** Even *without* seeding, the quantum muon **holds the
  DFT-relaxed (slightly expanded, +0.4 %) T-cage** and stays muonium — the muon
  and the lattice relaxation are mutually consistent. (Opposite of silicon, §3.2.)
- **Classical BC (#4, E = −90.730 Ha):** bond-centred muonium **+0.53** at
  ~2.6 bohr, corroborating the quantum BC result.

### 3.2 Silicon — diamagnetic at T, weak muonium only at BC

Silicon behaves oppositely at the T-site, and this is the study's central
contrast.

- **T-site is diamagnetic in every construction:** unrelaxed quantum (#1),
  DFT-relaxed quantum (#10), **T-seeded** quantum held inside the relaxed cage
  (#17), and even the **classical fixed-H** T-muon (#11) — all show
  `localized_excess ≈ 0` at all radii, no contact polarisation. The unpaired
  electron delocalises into the conduction bands rather than binding.
- **The muon avoids the relaxed T-cage unless forced (EXP-003).** DFT contracts
  the Si cage (−1.1 %) for a *classical* proton; the *quantum* muon with ZPM
  prefers the roomier unrelaxed cage and, unseeded, sits ~8.5 bohr away at an
  essentially-unrelaxed T-site. Seeding it inside the relaxed cage (#17, EXP-003b)
  makes it *hold* the cage — yet it is **still diamagnetic**. So the diamagnetic
  verdict is not a muon-site artefact.
- **BC binds — weakly.** The BC-seeded silicon muon (#15) holds BC and forms
  **weak** bond-centred muonium (excess **+0.25**, about half of diamond's, in a
  larger, more diffuse ~3 bohr shell still rising at the cell's edge). So "silicon
  never binds" is wrong — **silicon binding is site-dependent** (weak at BC, absent
  at T).

### 3.3 The off-BC trap — a sampling vs optimisation puzzle (EXP-001/004/005)

An **unseeded** diamond BC muon localises **off-BC** (in a T-like cage), not at
the bond centre. Is that physics or an artefact?

- **EXP-001 (classical PES, Quantum ESPRESSO):** BC is the **true global minimum,
  ~1.90 eV below off-T** — off-T is a higher local minimum. So the unseeded VMC
  result is a **trap**, not the true site. (This motivated the seeding experiments,
  EXP-002.)
- **EXP-004 (wide muon proposal):** keeping the muon's MCMC proposal wide through
  early training did **not** un-trap it — the adapter unwinds the width before
  basin selection. H1 falsified.
- **EXP-005 (frozen-wide adapter, maximum mobility):** even with the proposal
  **permanently wide**, the muon collapses to a non-BC cage with **0 % BC
  occupancy, ever.** ⇒ the off-BC trap is an **optimisation / representability**
  artefact of the ansatz, **not** a sampling-mobility problem. The lever is at the
  **envelope level**, not the proposal.

### 3.4 Muonium formation dynamics (EXP-006) — a striking decoupling

Sweeping checkpoints across training (step 0 → converged) for four runs revealed
how the bound state assembles, and one surprising result:

1. **Localisation precedes binding (~10–40k step lag).** The muon RMS collapses to
   its site by ~step 2000, but the net-spin excess is still ≈ 0 then; the electron
   binds only over steps ~6k–50k. **The muon finds its site first; the electron
   binds later.**
2. **The exchange-correlation hole forms first.** Early on the excess goes
   *negative* (spin-down depletion is carved), then the bound up-electron fills in
   (−ve → 0 → +plateau).
3. **Site sets the character, locked in from onset:** T = contact (rises from
   r = 0, contracts over training); BC = a fixed-radius shell with zero contact.
4. **★ Energy ⟂ local spin density.** Both spin channels — and the net-spin excess
   — **freeze by ~step 50k**, while the **energy keeps falling by 260–284 mHa** for
   hundreds of thousands more steps (the after-50k descent is ~24 % of the total
   energy drop). The classical fixed-origin control (#7) rules out an averaging
   artefact, so the decoupling is real: energy is a global/extensive bulk-
   correlation quantity that keeps improving; the muonium spin structure is
   intensive/local and saturates early.
   **Practical payoff:** muon-site and muonium verdicts are **valid on
   partially-trained (energy-unconverged) nets from ~50k steps** — which
   retro-validates the site/spin conclusions above.

### 3.5 Results at a glance

| # | Experiment | Verdict |
|---|---|---|
| 001 | off-T vs BC classical PES | **H0 rejected** — BC is the true minimum (1.90 eV below off-T); off-T is a trap |
| 002 | BC-seeded quantum muon | **BC HELD** — anisotropic (bond-centred) muonium; lowest-E quantum diamond (**most important result**) |
| 003 | Si T-relaxed bound state | **Diamagnetic**; unseeded muon avoids the relaxed cage (site caveat) |
| 003b | Si T-*seeded* + classical | **Diamagnetic** even holding the relaxed cage and for classical fixed-H — host-material contrast |
| 004 | wide muon proposal | H1 falsified — width unwinds before basin selection |
| 005 | frozen-wide adapter | **H0** — 0 % BC even permanently wide ⇒ trap is an optimisation artefact |
| 006 | energy vs local spin | **H0** — decoupled; spin structure frozen by ~50k while E falls 260–284 mHa; verdicts valid on un-converged nets |

---

## 4. Key takeaways

1. **Diamond = muonium at T (contact) and BC (bond-orbital); silicon T =
   diamagnetic, BC = weak muonium.** A genuine host-material contrast, robust to
   relaxation, seeding, and classical-vs-quantum muon treatment.
2. **Seeding is a legitimate and necessary tool.** Unseeded local VMC traps the
   heavy muon in the wrong basin (the muonium chicken-and-egg: no up-density at the
   muon early ⇒ gradients never learn to bind it). Seeding at the DFT/PES-preferred
   site, with an energy check, recovers the true state.
3. **The off-BC trap is an optimisation/representability limit, not a sampling
   one** — the next lever is the ansatz envelope, not the MCMC proposal.
4. **Muonium verdicts are readable on partially-trained nets** (spin structure
   freezes ~10× earlier than the energy), which makes the campaign much cheaper.

---

## 5. Open problem & next steps

**The one discrepancy with experiment.** Real µSR sees **normal muonium (Mu⁰_T)**
at the silicon T-site (isotropic, hyperfine ≈ 45 % of vacuum), plus anomalous Mu\*
at BC. We reproduce the (weak) BC state but **miss Mu_T entirely** — every silicon
T construction is diamagnetic. This is almost certainly a **methodological**
limitation, not physics, and closing it is the current frontier. Two coupled
suspects, both on the *electron* side (muon site is ruled out by EXP-003b):

- **Representability.** The PBC multiplicative envelope is a plane-wave (Bloch)
  form centred on the host **nuclei**; it has **no localised, decaying component**,
  and the quantum muon — being a walker particle, not an atom in `charges` — is
  **never an envelope centre.** A muon-localised orbital has to be built by the MLP
  fighting a plane-wave envelope with no anchoring basis function. Diamond's tight
  cage lets the surrounding nuclei's structure reach the T-site; silicon's roomy
  cage does not.
- **Finite size.** Silicon muonium is spatially large (high ε ≈ 11.7, light CB
  mass). Our one bound silicon state (BC, #15) already peaks at ~3 bohr and is
  still rising at the cell edge — a diffuse T-muonium electron in a 16-atom cell at
  Γ overlaps its periodic images and hybridises into the bands ⇒ reads diamagnetic.

**Planned experiments (designed, not yet run):**

- **EXP-008 — electron seeding (cheapest, do first).** Seed an up-spin walker on a
  **classical fixed** Si-T muon and watch whether the bound state **holds** or
  **washes out** — a clean discriminator: *holds & E ≤ diamagnetic baseline* ⇒ the
  blocker was **basin-selection** (a cheap win, no ansatz change); *washes out* ⇒
  **representability** ⇒ EXP-007. The classical vehicle isolates electron binding
  (muon pinned, already an envelope centre) and gives a matched-geometry energy
  test. Needs only new `mcmc.electron_init_coord/_width` plumbing.
- **EXP-007 — muon-anchored diffuse envelope + finite-size arm.** Add a diffuse,
  learnable, muon-centred decaying envelope term multiplied into the plane-wave
  PBC envelope (so bulk bands keep the Bloch form while a muon-localised orbital
  becomes representable), plus an electron–muon Jastrow cusp, and a 3×3×3 (54-atom)
  silicon-T cell to test whether the bound electron simply needs room. Success =
  a localised net-spin excess plateau at T with g↑/g↓ > 1 and a hyperfine
  consistent with the experimental ≈ 45 %-of-vacuum Mu_T, at energy ≤ the
  diamagnetic run.

**Other open threads:** extract the **contact density ρ_s(0)** / hyperfine with a
finer SRPD estimator (current 0.1-bohr bins are too coarse at r → 0) to compare
directly against µSR hyperfine constants; and confirm the diamond predictions
against experimental diamond muonium parameters.

---

## 6. Where the data & code live (post cluster expiry)

- **Code, configs, analysis tools, experiment notebook, and full research memory:**
  GitHub `Parvfect/ferminet_piku`, branch `muon_width`. `experiments/EXP-*.md` is
  the lab notebook; `claude_memory/` holds the detailed per-run results and energy
  tables.
- **Trained weights & inference data:** curated archive
  `parv_curated_backup_2026-08-17.tar.gz` (newest checkpoint per run + all energy
  trajectories + all inference position dumps + spin-density curves) plus a
  checkpoint supplement for the runs analysed directly from checkpoints. Full
  inventory, per-run system/weights map, and restore instructions in
  **`backup_details.md`**; backup process log in `data_backup.md`.
