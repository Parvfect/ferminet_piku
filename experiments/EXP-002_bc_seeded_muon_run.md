# EXP-002: BC-seeded quantum muon run — is off-T a trap?

- **Date:** 2026-06-24   **Status:** planned (awaiting verification before launch)
- **System:** diamond 2×2 supercell, BC-relaxed cage (`pp_relax_2` geometry), 16 C +
  1 muon, doublet. Identical to `pp_relax_2` in every respect **except the muon
  walker initialisation**.

## Question / Motivation (why)
EXP-001 proved (classically) that BC is the global muon minimum, **1.90 eV below**
the off-T site FermiNet localises at, and that the FermiNet result is a total trap
(0% of samples near BC). The *selection* mechanism is unresolved (EXP-001
"Unresolved"): init and forces are symmetric, yet the optimiser collapses 100% to
the higher off-T basin. This experiment tests the trap directly: **if we initialise
the muon walkers at BC, does the network hold the deeper BC basin, or drain to
off-T regardless?**

## Hypothesis (H1)
off-T is a metastable optimisation trap. Seeded at BC with a *fresh* network, the
muon stays localised at BC and the run converges to an energy **below** the off-T
run's −90.669 Ha. ⇒ off-T was a sampling artifact; BC is the true quantum site.

## Null hypothesis (H0)
BC seeding makes no lasting difference: the walkers drain to off-T during training
and the run reproduces the off-T site and ≈−90.669 Ha. ⇒ off-T is genuinely
preferred by the quantum optimiser (which would *contradict* the EXP-001 classical
PES and be highly significant — it would point to a large ZPE effect or a
representability bias).

## Alternative explanations (confounds to rule out before trusting a "drain")
- **Warm-start contamination** — if any pp_relax_2 checkpoint is restored, the
  off-T-peaked ψ is reintroduced and the walkers drain *for that reason*, not
  because BC is disfavoured. Must train from a fresh net (see risks).
- **Re-seeding on restart** — if walkers are re-initialised at BC on every job
  restart, the chain never equilibrates. Must seed only on first launch.
- **Wrong-particle seeding** — if the override hits an electron instead of the
  muon, the muon is left at its default carbon-centred init and the test is void.

## Method (experiment detail)
Fresh-network FermiNet training, carbons frozen at the BC-relaxed positions, muon
walkers initialised at the BC site. **Must be from scratch** — warm-starting the
off-T checkpoint reintroduces the off-T peak and the walkers drain within burn-in.
All other settings copied verbatim from `pp_relax_2` training.

### Code changes
**1. `ferminet/init.py` — `init_electrons` (~line 37).** Add two optional kwargs and,
*after* the atom-centred Gaussian (~lines 88–91), override only the last particle
(the muon; index `sum(electrons)-1` = 65 for particles `(33,32,1)`):
```python
def init_electrons(..., muon_init_coord=None, muon_init_width=0.5):
    ...                                   # existing electron_positions + Gaussian noise
    if muon_init_coord is not None:
        ep = electron_positions.reshape(batch_size, sum(electrons), ndim)
        key, subkey = jax.random.split(key)
        ep = ep.at[:, -1, :].set(jnp.asarray(muon_init_coord)
                                 + jax.random.normal(subkey, (batch_size, ndim)) * muon_init_width)
        electron_positions = ep.reshape(batch_size, -1)
```
Electrons keep their normal carbon-centred init; only the muon is moved.

**2. `ferminet/init.py` — `init_mcmc_data` (call at ~line 127).** Pass the config through:
```python
muon_init_coord = cfg.mcmc.muon_init_coord,
muon_init_width = cfg.mcmc.muon_init_width,
```

**3. `ferminet/base_config.py` — mcmc dict (~line 244).** Add defaults (keeps all
existing runs unchanged, since `None` preserves the old code path):
```python
'muon_init_coord': None,   # (x,y,z) bohr in the molecule frame; None = default init
'muon_init_width': 0.5,    # Gaussian width about muon_init_coord (bohr)
```

**4. New config `configs/diamond/bc_relaxed/bc_seeded.py`** (copy of the `pp_relax_2`
*training* config) + matching job script. Changes only:
```python
cfg.mcmc.muon_init_coord = (0.84282, 0.84282, 0.84282)   # relaxed BC midpoint, bohr
cfg.mcmc.muon_init_width = 0.3                            # tight at BC
# Fresh net + first-launch seeding: point restore at the (empty) save dir so
# find_last_checkpoint() returns None and the net trains from scratch.
cfg.log.save_path    = "/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_bc_seeded"
cfg.log.restore_path = cfg.log.save_path     # empty on first launch -> no restore
# Leave cfg.restart.load_data at its default True (see risks: False re-seeds every restart)
```
`muon_init_coord` is the midpoint of relaxed C0 (−0.05139·a) and C1 (0.30149·a),
a = 6.74 bohr ⇒ 0.125047·a = 0.84282 bohr per component (same frame as the molecule).

### Verification (before & during)
- **Before launch:** print the initial muon walker positions and confirm they are at
  BC (≈0.125 cubic-frac), while electrons remain carbon-centred. Confirm the log says
  "No checkpoint found. Training new model." (fresh net, no warm-start).
- **During:** periodically run short inference off a mid-training checkpoint and
  `muon_site_analysis.py` — peak at BC (0.125) or off-T (0.474)? Watch the VMC energy
  vs the off-T value −90.669 Ha.

## Result
_(pending)_

## Verdict
_(pending)_

## Things that could break (risks & mitigations)
1. **Warm-start contamination (highest risk).** `find_last_checkpoint(save_path) or
   find_last_checkpoint(restore_path)` will restore *any* checkpoint it finds. If
   `restore_path`/`save_path` resolve to a pp_relax_2 (off-T) checkpoint, the run
   restores the off-T net and "drains" for the wrong reason → false H0. *Mitigation:*
   brand-new empty `save_path`, `restore_path = save_path`; verify the
   "Training new model" log line.
2. **`load_data=False` re-seeds every restart (corrected from the original plan).**
   With `load_data=False`, each job restart re-runs `init_mcmc_data` → re-seeds the
   muon at BC every time, resetting the chain and discarding equilibration. First
   launch seeds regardless (no checkpoint exists yet), so we want **`load_data=True`
   (default)** so restarts *continue* the chain. *Mitigation:* do not set it to False.
3. **Wrong particle overridden.** The override assumes the muon is the last position
   (index `sum(particles)-1`), consistent with `particles=(33,32,1)` and the
   per-species charge/mass ordering. If ordering ever differs, an electron gets
   moved and the muon stays default. *Mitigation:* the pre-launch position dump (the
   moved particle must carry the muon, i.e. index 65) is the check.
4. **Early training instability / NaNs.** A muon clamped near BC sits inside the
   stretched-bond electron density → large local energies and variance spikes early,
   risking NaNs before the net adapts. *Mitigation:* `optim.reset_if_nan=True`,
   watch the first few k steps; consider a gentler LR warmup if it blows up.
5. **ConfigDict access.** `cfg.mcmc.muon_init_coord` must resolve (ml_collections).
   Since base_config defines it, direct attribute access is safe; avoid `.get()` if
   unsupported. A locked/Frozen ConfigDict could reject assignment in the config fn
   (it shouldn't, mirroring existing assignments).
6. **Unit / frame mismatch.** `muon_init_coord` must be **bohr cartesian** in the same
   frame as `cfg.system.molecule`. A fractional value (≈0.125) would seed the muon ~7×
   too close to the origin. *Mitigation:* derive from the same C0/C1 coords.
7. **Cost & continuity.** From-scratch ≈280k steps (days); needs the usual afterany
   chaining. A mid-run timeout/NaN that triggers a bad restore could silently convert
   the run into a warm-started (contaminated) one — re-check risk 1 after each restart.
8. **Interpreting a "drain".** If it drains to off-T, we must first rule out risks
   1–3 before concluding off-T is genuinely preferred. A clean drain (fresh net,
   correct particle, no contamination) would be the surprising, publishable outcome.
9. **PBC wrapping / multi-host.** Seeding + 0.3 bohr width stays inside the cell;
   the override is applied to the full host batch before reshape/sharding, matching
   the existing flow. Low risk, but confirm walker shapes are unchanged.
10. **Backward compatibility.** New kwargs default to `None`/old behaviour, so other
    `init_electrons` callers and existing configs are unaffected. Low risk.

## Next / current steps
- **User verifies** the code changes (diffs) and config before any submit.
- On approval: implement 1–4, run the pre-launch position-dump check, then launch.
- Consider launching **multiple-seed `pp_relax_2` reruns** in parallel (EXP-003?) to
  separate spontaneous symmetry breaking from a representability bias — complementary
  to this run. See EXP-001 "Unresolved".
- Cheaper alternative if a full retrain is too costly: warm-start + **harmonic
  restraint** tethering the muon at BC (needs a local-energy term), to read off BC's
  quantum energy without from-scratch training.
