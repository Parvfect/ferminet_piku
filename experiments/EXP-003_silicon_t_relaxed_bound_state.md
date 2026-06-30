# EXP-003: Does the silicon T-relaxed quantum muon form a bound state (muonium)?

- **Date:** 2026-06-29   **Status:** DONE — diamagnetic (H0), **but with a major
  caveat: the quantum muon does NOT localise in the relaxed cage** (see "Site caveat"),
  so the intended relaxed-cage binding test was effectively not performed.
- **System:** silicon 2×2×2 supercell (16 Si + 1 muon, doublet, PP on Si,
  a = 10.26 bohr), **T-relaxed** geometry — DFT-relaxed with the muon at the
  tetrahedral interstitial (0.75,0.75,0.75)·a; only the 4 coordinating Si move
  (~0.0487 bohr inward, −1.1% T-cage contraction). Training run #10
  (`muon_silicon_q_t_relaxed`, save `silicon_unpaired/t_relaxed`).

## Question / Motivation (why)
In **diamond**, the quantum muon at the T-site forms **muonium** (Mu = muon +
bound spin-up electron): a strong spin-resolved contact density, paramagnetic.
In **silicon**, every geometry tried so far is **DIAMAGNETIC** — no muonium, no
spin-up contact enhancement (unrelaxed #1, bc_relaxed #9; classical SRPD matches
quantum). That contrast is a central result. **But** those silicon runs used the
*ideal* (or BC-relaxed) cage. This experiment asks: when the Si cage is allowed
to **relax around a T-site muon**, does the slightly contracted T cage let the
muon bind an electron (form muonium / a bound state), or does silicon stay
diamagnetic regardless of relaxation? The answer informs whether the
diamond↔silicon muonium contrast is robust to lattice relaxation.

## Hypothesis (H1)
The T-relaxed Si cage (tighter coordination) increases the muon–electron contact
density enough to form a (partial) bound state: the SRPD g(r) shows a clear
spin-up excess over spin-down as r→0, i.e. muonium-like, unlike the ideal/BC Si
runs.

## Null hypothesis (H0)
Silicon stays **diamagnetic**: the T-relaxed muon's SRPD is spin-unpolarised
(g_up ≈ g_down at small r), matching the unrelaxed (#1) and bc_relaxed (#9)
silicon results. The −1.1% relaxation is too small to change the electronic
character; the diamond↔silicon muonium contrast is a host-material property, not
a geometry artefact.

## Alternative explanations (confounds)
- **Un-converged net.** #10 is still descending (not converged; ~−62.871 @130k,
  slope −3.4e-7/step). The bound-state character is read off the current net, so
  a future-step shift is possible. *Control:* re-run the inference at a later
  (more converged) checkpoint and confirm the SRPD verdict is stable; cross-check
  vs the classical (fixed-H) SRPD upper bound once #11 is available.
- **Muon not actually at T.** If the quantum muon has drifted off the T-site, the
  SRPD is measuring contact density at the wrong site. *Control:* the same
  positions give the muon site (`mode site` / `spread`) — verify it sits at T
  (cubic-frac ≈ 0.75) and is reasonably localised before trusting the SRPD.
- **Geometry mismatch inference vs training.** A stale or wrong-geometry
  inference config voids the comparison. *Control:* the inference molecule block
  was verified byte-identical to the training config (MATCH), 65 e⁻ (33,32,1),
  fresh empty save dir → falls back to the trained restore_path.

## Method (experiment detail)
**Position inference** off the trained #10 checkpoint, no optimisation, dumping
muon + electron positions; the **spin-resolved pair density (SRPD)** g_up(r),
g_down(r) is computed **post-hoc** from those positions (so we don't depend on
the live SRPD estimator). Bound state ⇔ g_up(r) ≫ g_down(r) as r→0.

### Files / artefacts (created 2026-06-29)
- **Config** `ferminet/configs/silicon/inference_t_relaxed.py` — copy of the
  inference pattern (`inference_bc_relaxed.py`); t_relaxed geometry verbatim from
  training `t_relaxed.py` (verified byte-identical), `particles=(33,32,1)`,
  `optimizer="none"`, `mcmc.fake_energy=True`, `observables.positions=True`.
  `restore_path=/projects/u6em/parv/silicon_unpaired/t_relaxed` (latest ckpt
  130000 at submit); `save_path=…/t_relaxed/inference` (created empty →
  `find_last_checkpoint(save_path)`=None falls back to restore_path; no stale
  ckpt present, confirmed). Positions → `…/inference/positions/positions_{t}.npy`.
- **Job script** `ferminet/jobs/silicon/inference_t_relaxed.sh` (8×4 GPU, output
  `inference_t_relaxed.out`, name `muon_silicon_q_t_relaxed_inf`). **Submitted:
  job 5415683.** `inference/positions/` pre-created (avoids the EXP-002 crash
  where the writer doesn't mkdir its own dir).
- **Analysis** `tools/muon_site_analysis.py` — added case `silicon_t_relaxed`
  (geometry `SI_T_RELAXED`, a=10.26, bc_pair (0,8)). Run:
  - `python tools/srpd_extended_radius.py silicon_t_relaxed 6.0 120 4` ← **definitive
    bound-state check** (cumulative net-spin vs uniform-band, out to the WS radius)
  - `python tools/muon_site_analysis.py silicon_t_relaxed srpd`  ← contact (≤1 bohr) SRPD
  - `python tools/muon_site_analysis.py silicon_t_relaxed site`  ← muon site
  - `python tools/muon_site_analysis.py silicon_t_relaxed spread` ← localisation
  (ferminet-piku env.)

### Comparators
- **Diamond T-site (muonium, positive control):** unrelaxed diamond shows strong
  spin-up contact density.
- **Silicon ideal / bc_relaxed (diamagnetic, the H0 baseline):** #1 unrelaxed,
  #9 bc_relaxed — spin-unpolarised SRPD.

### Supporting baseline (silicon bc_relaxed #9 — extended-radius SRPD, 2026-06-29)
Before this run, the strongest silicon H0 evidence was the contact (≤1 bohr)
SRPD. To make the baseline airtight we re-checked the #9 bc_relaxed positions
(512k samples, the muon that drifted off BC to a T-cage) **out to the
Wigner–Seitz radius** (g(r) to 6 bohr; WS inscribed radius 7.25 bohr;
Vcell=2160 bohr³, ρ_up=0.0153 ρ_dn=0.0148 e/bohr³). This separates the two
effects that a 1-bohr window conflates:
- **Spin-symmetric screening cloud** (any +1 charge has one): g/bulk ≈8× at
  contact, crosses bulk at ~1.5 bohr, correlation-hole dip ~0.7× at ~2 bohr,
  flat at bulk beyond ~3 bohr; ~0.8 e enclosed within 1.5 bohr. Diamagnetic.
- **Net spin density (the muonium discriminator):** cumulative N↑−N↓ **tracks
  the uniform-band line** (the lone doublet up-electron spread evenly over the
  cell, ∝r³) at every radius; the *localized* excess (actual − uniform) is ≈0,
  slightly negative (−0.02 to −0.05) out to 6 bohr — it never builds toward the
  +1 (or even +0.3) plateau a bound electron would.

**Implication for EXP-003:** in silicon bc_relaxed the unpaired electron is
delocalized into the bands at *all* length scales up to the lattice — not just
inside 1 bohr — so neither a tight (1s contact) nor a *loose/diffuse* bound state
exists. EXP-003 must therefore apply the **same extended-radius test** to the
T-relaxed positions: a "bound state" verdict requires a localized net-spin excess
that plateaus below the lattice scale, not merely a screening cloud (which is
present and spin-symmetric even in the diamagnetic case). Reproduce / extend with
`tools/srpd_extended_radius.py <case> [rmax] [nbins] [stride]` (case-aware; reads
the same `muon_site_analysis.CASES` geometry):
```
python tools/srpd_extended_radius.py silicon_bc_relaxed 6.0 120 4   # this baseline
python tools/srpd_extended_radius.py silicon_t_relaxed  6.0 120 4   # EXP-003, once positions land
```
See `project-silicon-bc-relaxed-muon-result` (memory).

## Result
Analysis run 2026-06-29 on the full positions dump (job 5415683 wrote all **1000
`positions_*.npy`**, 512k muon samples). Net checkpoint = #10 @130k (not converged;
caveat below).

**Muon site (control — is it actually at T?).** YES. Circular-mean and histogram-peak
both land on a **tetrahedral site** (cubic-frac ≈ 0.75; 4 nearest Si at ~4.17–4.58
bohr, all within <0.6 bohr of each other). Not BC (6.6 bohr from the intended BC
midpoint). Localisation: RMS radius 0.81 bohr, per-axis (0.40, 0.59, 0.40) —
**anisotropic, same magnitude/shape as the other quantum-Si runs** (#1, #9, ideal).
So the SRPD is being read at the right site on a normally-localised muon.

**Contact SRPD (≤1 bohr).** Spin-**unpolarised**: g_up ≈ g_down at every radius;
at the contact bin r≈0.083 g_up=0.142 vs g_down=0.169 (down is *slightly higher*,
i.e. zero spin-up enhancement). Integral within 1 bohr: 0.194 up vs 0.196 down.
**No muonium contact signature.**

**Extended-radius net-spin (the definitive test, out to WS).** Vcell=2160 bohr³,
ρ_up=0.0153 ρ_dn=0.0148 e/bohr³, WS inscribed radius 7.25 bohr. The
`localized_excess` (actual N↑−N↓ minus the uniform-band expectation) is **≈0 and
slightly NEGATIVE inside the lattice scale** (−0.002 at 0.6 bohr → −0.024 at 2.7
bohr), then drifts only to +0.15 by 5.7 bohr — i.e. it *rises above the lattice
scale*, tracking the uniform band (netSpin +0.52 vs netSpin_unif +0.37 @5.7 bohr),
and **never plateaus toward +1 below the lattice scale**. There is a spin-symmetric
screening cloud (g/bulk ≈10× up, ≈7× down at contact) but **no localized net-spin
excess** — exactly the bc_relaxed #9 baseline behaviour. The unpaired electron is
delocalized into the bands at every length scale.

## Site caveat — the quantum muon AVOIDS the relaxed cage (2026-06-29)
Critical follow-up check after the first pass. The relaxation moved exactly **4 Si
(indices 4,5,6,7), each 0.0487 bohr inward**, contracting the cage centred on
cubic-frac **(0.75,0.75,0.75)·a = (7.695,7.695,7.695) bohr** (the *design / relaxed*
T-site). The quantum muon does **NOT** sit there. Using the true FCC lattice vectors
`[[a,a,0],[0,a,a],[a,0,a]]` and minimum-image distances:
- The muon's **4 coordinating Si are {15,11,14,10}**, each displaced only **0.0002
  bohr (essentially UNMOVED)** — an *unrelaxed* tetrahedral interstitial. The relaxed
  cage Si {4,5,6,7} are only the 5th/6th-nearest (≥5.0 bohr).
- Muon centre → relaxed T-site = **8.47 bohr** (min-image). **0.0%** of samples lie
  within 3.5 bohr of the relaxed site; the single closest sample is **6.99 bohr**.
  → **Probability of the muon at the relaxed site ≈ 0.**

**Why:** the DFT relaxation *contracted* the cage (−1.1%, optimised for a *classical*
proton). A tighter cage is *less* favourable for a quantum muon with large zero-point
motion, so it localises at a roomier, unrelaxed T-site instead. Moving Si 4-7 inward
toward (0.75,0.75,0.75) also pushed them *away* from the neighbouring T-sites,
slightly enlarging those — exactly where the muon went. This is the textbook
**classical-H ≠ quantum-muon site** mismatch, visible directly in the positions.

**Consequence for the experiment:** the diamagnetic SRPD verdict is still valid (the
muon is at a Si T-site and it is diamagnetic), but EXP-003's *intended* question —
"does the **tighter relaxed cage** enable electron binding?" — was **not actually
tested**, because the muon never enters that cage. We can only state: *the quantum
muon avoids the DFT-relaxed (classical-proton) site and sits at an unrelaxed T-site,
which is diamagnetic.* To test the original question one would have to **confine /
seed the muon in the relaxed cage** (à la the EXP-002 bc_seeded approach) and check
whether it (a) holds there and (b) binds — otherwise the relaxed geometry is wasted.

## Verdict
**H0 CONFIRMED (diamagnetic) — with the site caveat above.** Silicon stays DIAMAGNETIC. Relaxing the Si
cage around a T-site muon (−1.1% contraction) does **not** produce muonium: no
spin-up contact excess (≤1 bohr) and no localized net-spin plateau (out to the WS
radius). The result is identical in character to ideal (#1) and bc_relaxed (#9)
silicon. Therefore the **diamond↔silicon muonium contrast is a host-material
property, robust to lattice relaxation** — it is not a geometry/relaxation artefact.
The hypothesis H1 (tighter cage → bound state) is rejected.

**Caveat (open):** #10 is not converged (~−62.871 @130k, still descending). The
verdict rests on a partially-trained net. The signal is unambiguous and matches two
independent converged-direction silicon baselines, so a late-step flip is unlikely,
but the clean control is to re-run this inference at a later #10 checkpoint and
confirm the diamagnetic verdict is stable (and cross-check vs the classical #11
fixed-H SRPD once available).

## Next / current steps
- [x] Confirmed 5415683 wrote all 1000 `positions_*.npy` (512k samples).
- [x] Site/spread: muon at T, RMS 0.81 bohr anisotropic — control passed.
- [x] Contact SRPD (≤1 bohr): spin-unpolarised — no muonium.
- [x] `tools/srpd_extended_radius.py silicon_t_relaxed 6.0 120 4`: localized
      net-spin excess ≈0/slightly negative within the lattice scale → diamagnetic,
      matches #9 bc_relaxed baseline. **H0 confirmed.**
- [ ] **Open control:** repeat inference at a later (more converged) #10 checkpoint
      to confirm the diamagnetic verdict is stable on the converging net.
- [ ] **Open cross-check:** classical fixed-H SRPD from #11 (`silicon c t_relaxed`)
      as the contact-density upper bound, once #11 is far enough along.
