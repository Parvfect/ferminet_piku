---
name: project-experiments
description: "Log of formal experiments (EXP-001 to EXP-003b) — hypotheses, implementations, and outcomes"
metadata: 
  node_type: memory
  type: project
  originSessionId: b3788c45-2950-4eff-a0df-720fbed90f1a
---

## EXP-001 — Classical PES along [111]: BC vs off-T energetics (DONE)

**Question:** Is off-T site truly preferred over BC, or is it an init trap?

**Method:** QE single-point forces (PBE, nspin2 doublet, frozen BC-relaxed C, H=classical muon) at 4 muon positions on [111] diagonal. Files in `~/qe/diamond_test/force_*.{in,out}` (login-node runs).

**Result:** **BC is the true global minimum, ~1.90 eV (70 mHa) BELOW off-T.** off-T is a higher LOCAL minimum. The watershed is C1: BC & off-T sit on opposite sides of C1 on [111]. Carbon-centred init + local MCMC traps quantum muon in off-T basin. 1.90 eV >> ZPE difference (~0.1–0.3 eV). **H0 REJECTED** — off-T is a trap, not the true site.

---

## EXP-002 — BC-seeded quantum muon (DONE; MOST IMPORTANT RESULT)

**Question:** If we seed MCMC walkers at BC, does the quantum muon HOLD BC?

**Implementation (branch `af`, commit `f14e142`, 2026-06-24):**
- `init.py init_electrons`: `muon_init_coord`/`muon_init_width` kwargs override only the last particle (muon); `None` default preserves existing runs
- `init.py init_mcmc_data`: passes `cfg.mcmc.muon_init_coord/width` through
- `base_config.py`: `muon_init_coord=None`, `muon_init_width=0.5` defaults
- `configs/diamond/bc_relaxed/bc_seeded.py`: muon seeded at BC (0.84282 bohr/comp ≈ 0.125a, width 0.3), fresh net
- `configs/silicon/bc_seeded.py` (#15, width 0.45 — wider by Si/C bond ratio ≈1.45)

**Diamond result (#14):** CONFIRMED — muon HOLDS BC at 0.037 bohr (176k net; was 0.30 bohr at 56k). Two C at 2.05/2.07 bohr ≈ half-bond. Oblate anisotropy (bond-pinned). loc_excess +0.47 e⁻ at ~2.7 bohr = **anisotropic (bond-centred) MUONIUM** — NOT diamagnetic as previously concluded from contact-only SRPD. See [[project-diamond-results]].

**Silicon result (#15):** Still in progress. Run blew up at 35–40k steps; restarted from ckpt_034000; now recovered and descending. Answer pending convergence.

---

## EXP-003 — Silicon T-relaxed bound-state test (DONE; H0 confirmed with major site caveat)

**Question:** Does silicon T-relaxed quantum muon form muonium (bound state)?

**Result:** DIAMAGNETIC (H0 confirmed). BUT muon did NOT localize in the DFT-relaxed cage — it sat at an UNRELAXED T-site (coordinating Si moved 0.0002 bohr vs design Si moved 0.0487 bohr), 8.47 bohr from the intended relaxed site. P(at relaxed site) ≈ 0.

**Why:** DFT contracted the Si cage (−1.1%) for a classical proton. Quantum muon with ZPM prefers roomier unrelaxed cage. Classical ≠ quantum site.

**Consequence:** Diamagnetic verdict valid but the intended "does tighter cage bind?" test was NOT performed. → Motivates EXP-003b.

---

## EXP-003b — Silicon T-seeded quantum muon (DONE 2026-07-08; H0 confirmed, diamagnetic)

**Question:** If seeded inside the relaxed T-cage, does the silicon muon HOLD it — and does the tighter cage let it bind an electron (muonium)?

**Implementation (job #17, submitted 2026-06-29):** `configs/silicon/t_seeded.py` — seeds muon at (0.75a)³ = relaxed T-site centroid, width 0.5, fresh net. Save `silicon_unpaired/t_seeded`.

**Result (analysed 2026-07-08, off training CHECKPOINTS — inference never dumped positions):**
- **Muon HOLDS the relaxed cage:** centre exactly (7.695³) bohr, RMS 0.78 bohr, isotropic (0.45³). Unlike unseeded #10 which fled 8.47 bohr. So the binding test is now actually performed.
- **Still DIAMAGNETIC (H0):** g↑≈g↓ at all radii; extended localized_excess ≈0/slightly −ve within lattice scale, only +0.08 @5.7 bohr, no plateau. Pooled last 20 ckpts (steps 197k–224k) = 81,920 configs.
- **Classical #11 cross-check (silicon_classical_t_relaxed, fixed-H upper bound):** also diamagnetic, contact ∫≤1 bohr 0.192 up / 0.199 dn. (Its training later diverged past 330k=`inf`; analysed the pre-blowup 2026-07-01 inference dump.)

**Verdict:** closes EXP-003's two open follow-ups. Diamond↔silicon muonium contrast is a **host-material property**, robust to relaxation / seeding / classical-vs-quantum. H1 rejected. Full writeup `experiments/EXP-003b_silicon_t_seeded_bound_state.md`.

**Tooling added:** `muon_site_analysis.iter_frames()` makes both tools **checkpoint-aware** (case key `checkpoints=` + `n_ckpt`, falls back to `data/positions` when no `positions_*.npy`) → bound-state analysis for any quantum run WITHOUT a fresh inference SLURM job. See [[project-silicon-results]], [[reference-tools]].

---

## EXP-004 / EXP-005 — is the off-BC trap sampling or optimisation? (DONE; H0)

**Question:** the unseeded diamond `bc_relaxed` muon localises off-BC (a T-cage). Is that a
*sampling* artifact (fixable by keeping the muon mobile) or an *optimisation/representability* one?

**EXP-004** (`pp_wide_burnin_v3`): one-shot wide muon init (`muon_move_width=0.3`). FALSIFIED H1 —
the adapter unwinds 0.3→0.072 by step 1800 before basin selection; muon localises at a T-cage 3.2 b
from BC. **EXP-005** (`pp_wide_burnin_frozen`, job 5534792): FREEZE the width wide all run
(`adapt_frequency=100000`, zero code). **H0 CONFIRMED** — even permanently wide, the muon collapses
to one non-BC T-cage by step ~500, **0% BC occupancy ever**. ⇒ off-BC is an OPTIMISATION artifact;
proposal mobility is not the lever. Next = **envelope-level muon bias**, not a slower adapter.
Full detail + the conceptual "why" in [[project-muon-mcmc-width-diffusion]] and
[[reference-vmc-basin-trapping-seeding]].

## QE relaxation setup (diamond BC, silicon BC)

Lives in `/home/u6em/parvfect.u6em/qe/diamond_test/`. See [[reference-qe-setup]] for full details.

**Key:** `bc_2_relax.in` / `bc_2_relax_unpaired.in` for diamond (nspin=2, tot_magnetization=1 matches FermiNet doublet 33↑/32↓). Silicon analogue: `si_bc_relax.in` / `si_bc_relax_unpaired.in`.

Geometry result: spin vs non-spin relax differ by max 0.006 Å (negligible); muon stays at BC in both; bc_relaxed geometry in FermiNet configs is already a fine approximation.
