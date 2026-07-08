# The localization–optimization interplay: why the FermiNet muon avoids the bond-centre

**Scope.** This note synthesises the diamond and silicon experiments that bear on
a single anomaly: the quantum muon in VMC/FermiNet **repeatedly localises away
from the bond-centre (BC) site even though BC is the proven global minimum.** It
draws together EXP-001 (classical PES), EXP-002 (BC seeding), EXP-003 (silicon
T-relaxed), and EXP-004/EXP-005 (the sampling-vs-optimisation isolation), and
argues that the off-BC result is an **optimisation / representability** failure of
the variational procedure, not a sampling-width artefact and not a statement about
the true physics.

Every quantitative claim below is traceable to a specific run and analysis tool
(`tools/muon_site_analysis.py`, `tools/srpd_extended_radius.py`,
`tools/muon_width_check.py`, `tools/muon_diffusion_check.py`) and to the
`experiments/EXP-00N_*.md` notebook entries. Units are **bohr / hartree /
electron-mass** unless stated. Assumptions are flagged inline as **[A]**.

---

## 1. The anomaly: BC is the global minimum, but the unseeded muon runs to an off-T cage

### 1.1 BC is the true minimum (classical ground truth, EXP-001)
On the DFT-relaxed diamond cage, a classical (clamped) muon placed along the
[111] diagonal gives, with only the muon moved and the carbons frozen at their
BC-relaxed positions (Quantum ESPRESSO, PBE, nspin=2 doublet):

| muon position (cubic-frac) | energy (Ry) | note |
|---|---|---|
| **BC (0.125)** | **−295.60317** | force ≈ 0, deep global min (= relaxation energy) |
| off-T (0.474) | −295.46363 | force ≈ 0, **local** min, **+0.1395 Ry = +1.90 eV ≈ +70 mHa** |
| 0.216 (BC-side of C1) | — | force −2.97·[111] → rolls **to BC** |
| 0.387 (T-side of C1) | — | force +2.95·[111] → rolls **to off-T** |

So BC sits **~1.90 eV (≈70 mHa) below** the off-T local minimum. The two basins
are separated by a **watershed at the C1 nucleus** on the [111] diagonal: a muon
cannot pass from one basin to the other without crossing through C1. This ordering
is far larger than any plausible zero-point-energy (ZPE) difference (~0.1–0.3 eV,
and off-T's greater openness helps it by at most that much), so **[A]** taking PBE
site-ordering as ground truth, BC is the muon's true site quantum-mechanically as
well. (EXP-001, verdict: *H0 rejected — BC is the true minimum; off-T is a trap*.)

### 1.2 The unseeded quantum muon localises off-BC — reproducibly
Running FermiNet with the default (symmetric, random-carbon-centred) muon
initialisation on that same BC-relaxed cage does **not** place the muon at BC:

- **Diamond `bc_relaxed/pp_relax_2` (#8):** the muon centre is **4.06 bohr from
  BC** and sits at an asymmetric off-T site — one carbon at ~2.0 bohr, the rest at
  2.7–3.1 bohr. **0.000 %** of 512k samples lie within 3.24 bohr of BC; 100 % at
  off-T (a total trap, no secondary peak). (`muon_site_analysis.py bc_relaxed_pp_relax_2`.)
- The trap is **not** an initialisation bias: replicating `init.py` (random carbon
  + 1-bohr Gaussian) puts **~1.70 %** of walkers within 1.5 bohr of BC vs **~1.66 %**
  within 1.5 bohr of off-T — the init is **symmetric**, not T-favouring. So init,
  forces (symmetric about C1), and energetics (favour BC) all fail to explain the
  100 % → off-T collapse. It is a **variational-optimisation** phenomenon.

This is the anomaly the rest of the document dissects. The framing we adopt (and
then test) is that muon-site selection is a **basin-selection / reachability**
problem, not a stability problem — VMC's only guarantees are the variational bound
`E[ψ] ≥ E₀` and convergence to a *local* minimum; it finds the global GS only when
the landscape is benign, which a near-degenerate, heavy-particle, multi-well
landscape is not.

---

## 2. Isolating the cause: sampling vs optimisation (EXP-004, EXP-005)

The off-BC result has two candidate explanations that must be separated before any
fix is chosen:

- **Sampling failure** — the Metropolis proposal is too narrow / the walkers too
  short-lived to *reach* BC while ψ is still plastic, so BC never accumulates
  gradient signal. **Fixable** by keeping the muon mobile.
- **Optimisation / representability failure** — even with unlimited muon mobility,
  the optimiser *chooses* a non-BC site on energetics, because BC's depth is a
  muon–electron (muonium) **correlation the fresh network does not represent**, so
  putting amplitude at BC yields no early gradient reward (a chicken-and-egg).
  **Not** fixable by sampling; needs a representation-level change.

### 2.1 The checkpoint audit that motivated EXP-004
A width/diffusion audit of the trapped `bc_relaxed/pp` run
(`tools/muon_width_check.py`, `tools/muon_diffusion_check.py`) established:

- **Steady-state width is healthy — the naïve width-collapse hypothesis is
  false.** The adapted muon proposal settles at **~0.24 bohr, ~2.85× the electron
  width**, ~54 % acceptance, across every run. Width tracks *acceptance*, not mass
  (the `1/√m ≈ 0.07` intuition is wrong: one muon in a smooth interstitial tolerates
  a bigger step than the cusped electrons).
- **The muon ensemble is diffuse only at init.** Walker spread goes ~5.5 bohr
  (random init) → **0.15 bohr by step ~2000** (collapsed ~37×) and never
  re-diffuses. The muon commits to one basin within the first ~2000 optimiser steps.
- **Mechanism: width-adaptation lag.** The defaults are `move_width=0.02`,
  `adapt_frequency=100`, and the width does **not** adapt during burn-in, so through
  the first ~2000 steps the muon proposal is stuck at ~0.02–0.08 bohr — too small to
  hop basins — *exactly while KFAC localises ψ fastest*. **[A]** basin selection
  completes inside roughly the first ~2000–3400 optimiser steps (the "selection
  window"), inferred from the diffusion trajectory.

This made "off-BC = an early-time sampling artefact" a concrete, falsifiable
hypothesis — and it is precisely what EXP-004/005 test.

### 2.2 EXP-004 — a one-shot wide muon init does **not** un-trap the muon
A new per-species knob `cfg.mcmc.muon_move_width` was added (default `None` → byte-identical to
stock behaviour; when set, the **muon only** starts wide, electrons keep
`move_width`, each species adapts independently). EXP-004 runs the BC-relaxed cage,
**unseeded** (`muon_init_coord=None`), with `muon_move_width=0.3` and
`burn_in=2000` (ψ frozen during burn-in → pure equilibration to `|ψ_fresh|²`).

**The intervention fired correctly.** Step-0 checkpoint (`muon_width_check.py`,
`muon_diffusion_check.py`): muon width **0.3**, electrons **0.02**; muon walkers a
broad diffuse cloud, RMS radius ~12.7 bohr (≈ the free-diffusion prediction
`0.3·√(2·2000·p_move) ≈ 14.7`), only ~5 % of walkers within 3.2 bohr of BC — spread
everywhere, pinned nowhere. This is the correctly un-trapped precondition for the
sampling hypothesis.

**But it re-collapsed before basin selection completed (H1 falsified).** At step
10 200 of the valid `pp_wide_burnin_v3` run:

- The adaptive width unwound the wide init: **0.30 (step 0) → 0.072 (step 1800)** —
  briefly *narrower* than the electrons — because 0.3 bohr gives sub-50 % muon
  acceptance, so the adapter divided by 1.1 at nearly every early event. The
  `×/÷1.1-every-100-steps` cadence is the culprit.
- Ensemble diffuse only at init (spread 2.6–3.1 bohr, steps 0–400) → **~0.15 bohr by
  step ~700**. **BC occupancy drains 5.5 % → 0 % by step 300** and stays 0 %.
- Site: a **tetrahedral interstitial 3.2 bohr from BC** (0 % of walkers within 2
  bohr of BC), 4 carbons at ~2.6–3.0 bohr — a *different* T-cage from the original
  off-T, not BC. Energy −89.41 ± 0.12 (block-avg, step ≥8000), still above the off-T
  reference −90.669 and still descending.

**Verdict (EXP-004): proposal width alone is not the lever.** A one-shot wide init
is unwound by the adapter faster than the basin-selection window, and — critically —
keeping the walkers mobile for a while installs *no BC-preferring force on ψ*. This
is consistent with, but does not yet prove, the optimisation explanation.

### 2.3 EXP-005 — the decisive limit: freeze the proposal wide for the whole run
Slowing the adapter can never be conclusive (there is always a slower setting → a
still-off-BC result is goalpost-moving). The only informative single run is the
**limit**: freeze the width wide for the entire selection window and see what the
optimiser does with a *permanently* mobile muon. EXP-005 sets
`adapt_frequency=100000` (the guard `t>0 and t%freq==0` never fires before step
100 000 — 20× past the selection window; `100000` not `1e9` because `pmoves` is
allocated `(nspecies, adapt_frequency)`, so 1e9 → ~24 GB OOM), with
`muon_move_width=0.3`, electrons frozen at `move_width=0.07` (≈ their adapted
equilibrium). Unbiasedness is not a concern: **Metropolis samples the true `|ψ|²`
for any proposal width — width affects variance, never correctness.**

**Result (H0 confirmed).** Freeze verified: width flat `[0.07, 0.07, 0.3]` for all
34 checkpoints (`muon_width_check.py`) — the adapter never fired. Yet despite the
permanently-wide proposal:

- The cloud collapses **2.78 bohr (step 0) → ~0.14–0.30 bohr by step ~500**, then
  flat, into **one non-BC cage**.
- **0.0 % of walkers within 2 bohr of BC at *every* checkpoint, including step 0.**
- The collapse is complete by step ~500 **while acceptance was still healthy
  (~0.15) and the cloud 2–3 bohr wide** ⇒ it is **ψ-driven, not a low-acceptance
  freeze**.
- Site (folded to a common periodic image): a single stable **T-type open
  interstitial** at prim-frac ≈ (0.845, 0.577, 0.330), drifting ≤0.10 bohr from step
  300 to 3300, nearest carbons ~2.3–2.65 bohr, **~5 bohr from ideal BC**, 1.9 bohr
  off the nearest C–C bond midpoint (**not** a bond-centre). This reproduces
  EXP-004's T-cage **from an independent fresh net**.

**Verdict (EXP-005): the off-BC trap is an optimisation / representability artefact,
not a sampling one.** With a permanently maximally-mobile muon, ψ still evacuates
the muon into a non-BC cage and holds it there. **Proposal mobility is not the
lever; the sampling-width axis is exhausted.** The remaining lever is a
**representation-level** change (an envelope-level muon bias that shapes `ψ_fresh`
toward BC directly), or explicit seeding.

> **Caveat [A].** EXP-005 delivered a fixed *proposal*, not sustained realised
> *mobility*: once ψ sharpens the cage to ~0.3 bohr, a fixed 0.3-bohr proposal
> overshoots and muon acceptance craters to ~8 % late in the run. This does not
> weaken H0 — the collapse is complete by step ~500 while acceptance was still ~0.15
> and the cloud 2–3 bohr wide, with 0 % BC weight even then — but a fully airtight
> "sustained mobility" test would need a proposal that *tracks* the shrinking site
> (or a site-hop proposal). That refinement is only worth doing if the envelope-bias
> route also fails.

---

## 3. BC seeding: proving BC is a real, holdable basin the optimiser simply doesn't select (EXP-002)

If the failure is that the optimiser never *reaches* the BC basin, then supplying
the amplitude by hand — seeding the muon's **MCMC initial condition** at BC — should
break the chicken-and-egg. It does.

**Seeding is not cheating [A].** Seeding sets the walker *initial condition* only.
It does not touch the Hamiltonian, add a constraint or bias potential, or change
what Metropolis samples (still the true `|ψ|²`). It changes the optimisation *path*,
not the *objective*; the variational bound `E[ψ] ≥ E₀` holds for any ψ regardless of
how it was obtained, so a seeded-converged energy is a rigorous upper bound. After
seeding, the muon is fully quantum and free to leave — that it *holds* BC is a
**result**, not an imposition. (Contrast a clamped/classical muon, which genuinely
changes the physics by removing zero-point motion.)

**Diamond `bc_seeded` (EXP-002) — the muon HOLDS BC**, confirmed five independent
ways on the well-trained (~176k) net:

- **Position:** circular mean **0.037 bohr from BC** (0.30 bohr on the earlier 56k
  net — better training pins it dead-on).
- **Coordination:** two carbons at 2.05 / 2.07 bohr ≈ half-bond, then a 1.6-bohr gap
  — a true symmetric bond-centre.
- **Anisotropy:** the density is **oblate** — tight along the [111] bond (par RMS
  0.231) and wider perpendicular (perp RMS 0.481), the BC confinement signature (a T
  interstitial would be ~isotropic). The extracted bond axis is exactly [111].
- **SRPD** matches the classical fixed-BC run almost bin-for-bin.
- **Net spin (extended-radius):** a localised excess of **~+0.47 e⁻ at the bond
  scale (~2.7 bohr)** — anisotropic, bond-centred **muonium** (the classical
  clamped-BC run shows the same ~+0.53, so this is an electronic-structure property
  of BC, not a ZPM artefact).

**The decisive contrast — same geometry, only the init differs:**

| measure | `bc_seeded` (seeded @ BC) | `pp_relax_2` #8 (unseeded) |
|---|---|---|
| distance to BC | **0.30 bohr** | **4.06 bohr** |
| nearest carbons | 2.02 / 2.21 (both ≈ half-bond) | 6.14 / 2.05 (one carbon close) |
| anisotropy par/perp | 0.48 (oblate, on bond) | 9.88 (off-bond by a single C) |
| site | **true BC** | off-T trap |

Seeding the muon at BC makes it **hold** BC; the default init falls into the off-T
trap. Because seeding changes only the initial condition, this is direct evidence
that **BC is a genuine, holdable basin that the unseeded optimiser fails to select**
— the "reachability, not stability" reading. Seed-dependence of the converged energy
*is itself* the diagnostic of a basin-trapped landscape, and the lowest-energy seed
is the best GS estimate (basin-hopping by hand).

**The variational energy — the seed-independent arbiter — now confirms BC (the
strongest single piece of evidence).** A lower ⟨H⟩ is a strictly better ground-state
estimate *regardless of how the walkers were initialised*, so the energy comparison
between two seeded endpoints on the same Hamiltonian is not circular — it is the one
seed-free test. Block-averaged plateau energies (`tools/energy_convergence.py`, same
BC-relaxed cage, same (33,32,1)), refreshed 2026-07-08:

| run | E (E_h) | step | status |
|---|---|---|---|
| **BC-seeded #14 (seeded @ BC)** | **−90.69597 ± 0.00019** | 580k | **CONVERGED** (slope flat, drift < tol) |
| off-T #8 (unseeded) | −90.66904 ± 0.00029 | 280k | frozen |
| classical fixed-T #7 (no ZPM) | −90.69624 ± 0.00015 | 472k | converged |
| classical fixed-BC #4 (no ZPM) | −90.73010 ± 0.00029 | 338k | converged |

The BC-seeded quantum muon is now **converged ~26.9 mHa below the unseeded off-T
run** and is the **lowest quantum diamond run**, statistically tied with the
classical fixed-T reference (Δ ≈ 0.3 mHa, within SEM). This closes the earlier
ambiguity (at ~176k both were within ~14 mHa and neither converged): the variational
principle — the foundation VMC rests on — now rules for BC, independently of the
seed. It also confirms the classical EXP-001 ordering (§1.1) *within the quantum
theory*, so BC-below-off-T no longer rests on the assumption that muon ZPE is small.

**Silicon replicates the diamond seeding result.** The silicon `bc_seeded` run (#15)
pins the muon at BC (**0.18 bohr from ideal BC**, two Si at ~2.94/3.06 bohr ≈
half-bond), whereas the *unseeded* silicon `bc_relaxed` muon drifts ~4.4 bohr to a
near-symmetric T cage. So the unseeded-misses-BC / seeded-holds-BC pattern is **not
diamond-specific** — it is a property of the optimisation on this class of landscape.
(Silicon BC carries only a **weak** partial muonium, ~+0.26 e⁻ vs diamond's ~+0.47;
silicon muonium is site-dependent — diamagnetic at T, weak at BC — which is a
separate electronic-structure finding.)

---

## 4. Silicon T-relaxed: the muon flees the relaxed cage — the gradient never feels the well's depth (EXP-003)

The silicon T-relaxed run is the cleanest independent demonstration that **the
optimiser's gradient does not register a nearby, deeper well the muon is not already
sampling** — the same mechanism as the BC miss, seen on a different geometry.

Setup: the Si cage was DFT-relaxed **around a classical proton at the T-site**,
which moves the 4 coordinating Si **0.0487 bohr inward (−1.1 % contraction)** — a
tighter cage optimised for a point charge. The quantum muon was then run in that
relaxed cage, unseeded.

**The quantum muon does not go to the relaxed site.** Using the true FCC lattice
vectors and minimum-image distances (`muon_site_analysis.py silicon_t_relaxed`):

- The muon's 4 coordinating Si are a *different* set, each displaced only **0.0002
  bohr — essentially unmoved** (an **unrelaxed** T interstitial). The relaxed-cage Si
  are only 5th/6th-nearest (≥5.0 bohr).
- Muon centre → relaxed T-site = **8.47 bohr**. **0.0 %** of samples lie within 3.5
  bohr of the relaxed site; the single closest sample is 6.99 bohr. **P(muon at
  relaxed site) ≈ 0.**

**Why this is exactly the localization–optimization interplay.** With the muon's
amplitude already committed to the unrelaxed cage, the optimiser has **no mechanism
to discover that a lower-energy configuration exists a cage away**: **zero samples
reach the relaxed cage → zero gradient signal there → ψ → 0 outside the occupied
cage**. The gradient is computed from `∇E` evaluated on the *current* sample
distribution, so a well the walkers do not visit is invisible to it — it never
"feels the depth" of the alternative well. This is the identical failure to the BC
miss (§2), now demonstrated where the alternative site is a *relaxed* cage rather
than BC, so it cannot be dismissed as a BC-specific quirk. (EXP-003 verdict: the
intended relaxed-cage binding test was **not actually performed** because the muon
never enters that cage — the site mismatch is itself the finding.)

**The T-seeded run (#17) settles it: the flight was a trap, not a zero-point
preference (EXP-003b).** A tempting alternative reading is that the quantum muon
*correctly rejects* the contracted relaxed cage because a tighter well costs more
zero-point energy — i.e. the flight is physics, not a trap. Seeding the muon into
the relaxed cage falsifies that. Block-averaged plateau energies
(`tools/energy_convergence.py`, silicon, refreshed 2026-07-08):

| run | E (E_h) | step | status |
|---|---|---|---|
| **T-seeded #17 (seeded in the relaxed cage)** | **−62.89124 ± 0.00051** | 224k | noise-limited (slope flat) |
| T-relaxed #10 (unseeded, fled to unrelaxed cage) | −62.87392 ± 0.00034 | 138k | frozen |
| BC-seeded #15 | −62.89170 ± 0.00022 | 196k | descending |
| unrelaxed #1 / bc_relaxed #9 (unseeded, both T-cage) | −62.862 / −62.864 | — | frozen |

Seeding the muon **into** the contracted relaxed cage yields **~17.3 mHa *lower*
energy** than the unseeded run that fled it — the relaxed cage is genuinely the
better (lower total-energy) site, and the electronic relaxation gain **outweighs**
the extra zero-point cost. So the unseeded flight was **another basin-selection
trap**, precisely parallel to the diamond off-T miss — not a ZPM-driven expulsion.
(A ZPE *penalty* for the tighter well is real in direction but sub-dominant: it does
not reorder the sites.) This is the second geometry — after diamond BC (§3) — where
seeding recovers a converged/near-converged energy *below* what unseeded sampling
finds, which is strong evidence the effect is a property of the optimisation, not a
coincidence.

**The complementary control — diamond T-relaxed (#12) HOLDS its cage.** There the
DFT relaxation **expanded** the T-cage (+0.017 bohr, +0.4 %), and the unseeded
quantum muon **stays** dead-centre in it. So the pattern is not "quantum muons avoid
tight cages" (falsified by #17 above) but the plainer "the muon localises wherever
it is *already* sampling, and the optimiser will not move it to a better site it is
not sampling — regardless of that site's energy." Diamond-holds happens because the
default init already lands the muon in that (expanded) cage; silicon-flees happens
because the default init lands it elsewhere and it never revisits the relaxed cage.
Same "reachability governs the outcome, not energy ordering" conclusion as §2–§3.

---

## 5. Why a T-site is preferred in the *initial* optimisation steps

The above establishes *that* the optimiser selects a T-type cage over BC. This
section argues *why* the selection lands on T specifically, from the structure of a
**fresh** FermiNet ansatz and the VMC update — the mechanism that EXP-004/005 then
confirmed is decisive within the first few hundred steps. **[A]** this account rests
on reading `ψ_fresh` from `envelopes.py`/`init.py` at initialisation and on the
selection window being early (§2.1); it is a mechanistic argument corroborated by the
0 %-BC-occupancy-from-step-0 observations, not a separately isolated experiment.

**1. `ψ_fresh` is set by the envelope, and it peaks on the host atoms — not at any
interstitial.** At initialisation the network factor is a smooth O(1) modulation, so
the shape of `ψ_fresh` is the isotropic multiplicative envelope
`Σ_carbons π·exp(−σ·r)` with `π = σ = 1` (`envelopes.py`). Thus `|ψ_fresh|²` **peaks
on the carbons** and decays ~1 bohr into the interstitial. The muon is *enveloped by*
the host atoms (it is not itself an envelope centre). Crucially, in the interstitial
tails `|ψ_fresh|²` is, if anything, *mildly larger at BC* (~2 carbons at ~1.5 bohr)
than at off-T (~4 carbons at ~2.9 bohr). **So neither the (symmetric) init nor
`ψ_fresh` favours T by amplitude, and energetics favour BC — T is favoured by
neither.**

**2. T wins by *kinetic accessibility* during the evacuation off the host atoms.**
Because `ψ_fresh` puts the muon where the local energy `E_L` is huge (on the nuclear
cusps of the carbons), the first thing optimisation does is **evacuate** the muon
into the interstitial to lower `E_L`. Basin selection happens *during* this
evacuation. With small local Metropolis moves (~0.02–0.08 bohr early, per §2.1), the
only interstitial region reachable from a carbon is the **open tetrahedral cage → T**.
Reaching BC instead means moving *toward and through the C1 carbon watershed*
(EXP-001), where proposals land on/near a nuclear cusp and are rejected. So the muon
is **channelled into the T-cage regardless of BC being deeper** — the small early
step size makes T reachable and BC not.

**3. T is "electrostatically easy"; BC requires a correlation the fresh net has not
built.** A T-cage muon is stabilised by simple electrostatics (a +1 charge in an open
interstitial with a spin-symmetric screening cloud), which a fresh net represents
immediately. BC's *depth*, by contrast, is a **muonium correlation** — the muon binds
a specific spin-up electron into a bond-orbital (the ~+0.47 e⁻ localised excess seen
only once BC is seeded, §3). A fresh network **does not represent that correlation**,
so even the small BC-leaning amplitude in `ψ_fresh` yields **no low local energy and
hence no gradient reward** at BC in the early steps: `∇E` at step 0 points *away*
from BC (chicken-and-egg). The optimiser therefore has both a **kinetic** reason
(only T is reachable by small moves) and an **energetic-signal** reason (only T pays
off immediately) to commit to T.

**4. Mode collapse then locks it in — and the heavy mass makes it irreversible.**
Once amplitude commits to the T-cage, 0 samples reach BC → 0 gradient signal at BC →
ψ → 0 outside the cage → the deeper well becomes invisible (this is the mechanism
directly observed as **BC occupancy 5.5 % → 0 % by step 300** in EXP-004 and **0 %
from step 0** in EXP-005). And because the muon is a **heavy, distinguishable
particle (207× the electron mass)**, its eigenstates are tightly localised and
inter-cage tunnelling is exponentially suppressed — so even a later wide proposal
cannot hop it between cages (cross-cell acceptance ≈ 0, exactly why frozen-wide
EXP-005 did not help). The commitment made in the first few hundred steps is, in
practice, permanent.

This is the same class of hard problem as polaron self-trapping / Anderson
localisation / defect levels: a near-degenerate multi-well landscape (BC-vs-T is
meV–tens-of-meV on top of ~90 Ha, so the distinguishing gradient is buried in
sampling noise), a heavy particle that both localises ψ and blocks MCMC hopping, and
a stabilising feature (muonium) that the ansatz must *learn* before it can be
rewarded for it.

---

## 6. Synthesis and open items

**What is established.**

1. **BC is the true muon site**, on two independent yardsticks: classically (EXP-001:
   1.90 eV below off-T) and now in the **quantum VMC energy itself** — the converged
   BC-seeded run is **26.9 mHa below** the unseeded off-T run and the lowest quantum
   diamond run (§3). Yet the unseeded FermiNet muon reproducibly localises at a T-type
   interstitial (§1).
2. **BC is a real, holdable, lower-energy basin the unseeded optimiser fails to
   select.** Seeding the init at BC makes the fully-quantum muon hold BC in both
   diamond (0.037 bohr) and silicon (0.18 bohr), and — the seed-independent point — the
   **variational energy of the seeded run is lower**: diamond BC-seeded −90.696
   (converged) vs off-T −90.669; silicon T-seeded −62.891 vs unseeded-fled −62.874
   (§3, §4). Lower ⟨H⟩ is a better ground state regardless of init, so this is not
   circular — it is the one seed-free arbiter, and it favours the seeded site twice.
3. **The cause is (mostly) optimisation / representability, not proposal width.** A
   permanently maximally-mobile muon still misses BC (EXP-005: 0 % BC occupancy at
   every checkpoint) — the *proposal-width* axis is exhausted (§2). See the caveat
   below: this does **not** rule out an ergodicity/barrier-crossing sampling fix.
4. **The gradient cannot feel a deeper well the muon is not already sampling.**
   Silicon T-relaxed: the unseeded quantum muon flees the relaxed cage by 8.47 bohr
   (P ≈ 0 at the relaxed site), yet seeding it *into* that cage converges 17.3 mHa
   **lower** — so the flight was a basin-selection trap, not a zero-point preference
   for the roomier cage (§4).
5. **T is selected early** by kinetic accessibility (small moves reach only the open
   cage, BC sits behind the C1 watershed) plus the absence of an early gradient
   reward for the unlearned muonium correlation; mode collapse and the heavy mass
   then lock it in (§5).

**The remaining levers** are (a) representation-level — an **envelope-level muon
bias** that shapes `ψ_fresh` toward BC so visiting BC yields a gradient reward from
step 0 (the test that cleanly separates "cannot *sample* BC" from "cannot *represent*
BC"; explicit seeding is the initial-condition analogue, this seeds the
*representation*); and (b) sampling-level but **beyond proposal width** — a
**barrier-crossing sampler** (parallel tempering / replica exchange / a collective
mode-jump move). Local-move Metropolis, wide or narrow, cannot cross the C1 watershed
(cross-cell acceptance ≈ 0), so EXP-004/005 only exhausted *local* mobility; a
sampler that can visit BC from a symmetric start, and *chooses* it unaided, is the
clean demonstration that would upgrade "we found BC by seeding" to "the method finds
BC on its own." That test has not yet been run.

**Assumptions / caveats to keep in view.**

- **[A] PBE site-ordering as ground truth** (§1.1) is now *corroborated* by the
  quantum VMC energy (BC-seeded converged below off-T), so the old worry that muon
  ZPE (~0.1–0.3 eV) might overturn the classical gap is retired for diamond BC; the
  silicon T-seeded result likewise shows the ZPE penalty for a tighter cage is
  sub-dominant. ZPE reweighting is real in direction but does not reorder these sites.
- **[A] The "optimisation not sampling" split is not airtight.** EXP-005 froze the
  *proposal* wide but the muon cannot tunnel between cages under any local move, so an
  ergodicity-broken sampler starves the gradient in a way that mimics an optimisation
  choice. Only a barrier-crossing sampler (above) can fully separate the two — a
  point conceded to the sceptic. What EXP-005 *does* establish is narrower but solid:
  **wider local proposal width is not the lever.**
- **[A] The selection window is early (~first few hundred to ~2000 steps)** — inferred
  from the diffusion trajectory (§2.1), and consistent with the 0 %-BC-by-step-300/500
  observations, but not independently timed per run.
- **[A] Convergence.** Diamond BC-seeded #14 is now *converged* (−90.69597 ± 0.00019,
  flat slope), so the diamond energy verdict is solid. Silicon T-seeded #17
  (noise-limited) and BC-seeded #15 (still descending) are near-plateau upper bounds;
  EXP-004's wide-burnin run was still descending at −89.41. Site/spin *signatures* are
  unambiguous and reproduced across independent nets.
- **[A] Global minimum ≠ the µSR-observed site.** Even a perfect optimiser returns the
  *ground-state* site, whereas µSR sees kinetically-trapped *metastable* sites that
  coexist (e.g. Mu_BC and Mu_T in Si). Enumerating the metastable manifold — not just
  the winner — is a separate goal that also needs multi-site / barrier-crossing sampling.
- **[A] EXP-005's frozen proposal is not sustained realised mobility** once ψ localises
  (§2.3 caveat) — a site-tracking/hop proposal would close this.
- **[A] Single-seed runs.** Spontaneous symmetry breaking / seed-dependence of *which*
  T-cage is chosen is not fully ruled out (EXP-001's open multi-seed question). The
  *class* of outcome (T over BC) is robust across independent fresh nets; the specific
  cage may be seed-dependent.
- Most runs **clamp the host lattice** (only the muon is quantum); this is appropriate
  for the site-selection question but means the cages are not re-relaxed around where
  the muon actually goes.

---

*Sources: `experiments/EXP-001_muon_offT_vs_bc_classical.md`,
`EXP-002_bc_seeded_muon_run.md`, `EXP-003_silicon_t_relaxed_bound_state.md`,
`EXP-004_muon_width_burnin_diffusion.md`, `EXP-005_frozen_adapter_muon_mobility.md`;
analysis via `tools/muon_site_analysis.py`, `tools/srpd_extended_radius.py`,
`tools/muon_width_check.py`, `tools/muon_diffusion_check.py`.*
