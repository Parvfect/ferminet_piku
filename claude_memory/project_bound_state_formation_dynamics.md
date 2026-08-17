---
name: project-bound-state-formation-dynamics
description: "★ How muonium forms OVER TRAINING (checkpoint sweep): localization precedes binding; correlation hole first; T=contact vs BC=bond-centred; Si site-dependent"
metadata:
  node_type: memory
  type: project
  originSessionId: 9b5d594f-a580-446b-b49d-b4cc04892f69
---

## Muonium bound-state formation dynamics across training — 2026-07-08 ★

Swept training CHECKPOINTS (step 0→converged, ~19 steps each) of 4 runs, reading
the 4096 live walkers straight from each `qmcjax_ckpt_*.npz` (`data/positions`),
and per step computed muon localization (RMS spread about fcc-hist peak) +
**radius-resolved net-spin excess** `localized_excess(r)` = cumulative
N↑(r)−N↓(r) − uniform-band baseline, MOVING origin (electrons − each walker's own
muon). No inference dumps needed. Tool: scratchpad `bound_state_formation.py`
(reuses [[reference-tools]] `iter_frames`/geometry; radius-resolved is ESSENTIAL —
contact-only misses bond-centred BC muonium, see [[reference-tools]] gotcha).

**Converged peak net-spin excess (muonium strength) by run:**
| Run | muonium | peak excess | peak radius | contact ratio g↑/g↓ | character |
|---|---|---|---|---|---|
| diamond unrelaxed T (#6, unseeded) | YES | +0.42 | ~1.8 b (contracts 2.9→1.8) | 4.5–6.9 | **atomic/contact** (e⁻ on muon) |
| diamond BC-seeded (#14) | YES | +0.47 | ~2.6 b (fixed) | ~1.0 | **bond-centred** (0 contact) |
| silicon BC-seeded (#15) | WEAK | +0.22–0.28 | ~2.9–3.4 b (diffuse, still rising) | ~1.0 | **weak bond-centred** (~½ diamond) |
| silicon unrelaxed T (#1, control) | NO | ~0 | — | ~1.0 | diamagnetic |

### Findings (each a distinct result)
1. **Localization PRECEDES binding (~10–40k step lag).** Muon RMS collapses by
   step ~2000 (diamond-T 4.83→0.50 b; Si-T 7.46→0.46 b) but net-spin excess is
   still ≈0 then. Excess builds over step ~6k–50k. Muon finds its SITE first, the
   electron binds later.
2. **Even a PINNED muon takes ~50–75k steps to bind.** Diamond/silicon BC are
   SEEDED (muon at BC from step 0, d_site const) yet the bond-centred shell only
   reaches plateau by ~50–75k → the electronic bound state is a property of the
   OPTIMISING WAVEFUNCTION, not the muon position.
3. **The exchange-correlation HOLE forms before the bound state.** At step 2k–4k
   both diamond runs show NEGATIVE excess (profiles dip to −0.4/−0.6); spin-down
   depletion is carved first, then the bound up-electron fills in (excess climbs
   −ve → 0 → +plateau).
4. **Site sets the character, locked in from onset of binding.** T: excess rises
   from r=0, strong contact ratio, cloud CONTRACTS over training (diffuse→atomic).
   BC: excess is a SHELL with ZERO contact enhancement, radius fixed by bond
   geometry. A contact-only probe calls BC "diamagnetic" (historical error).
5. **★ Silicon binding is SITE-DEPENDENT, not absent.** Si is diamagnetic at T
   (T #1, t_relaxed, t_seeded all ~0) but forms WEAK bond-centred muonium at BC
   (~+0.25, ½ of diamond, larger/more diffuse ~2.9–3.4 b shell). So "silicon never
   binds" is WRONG — it binds weakly only at BC. Confirms + independently
   reproduces [[project-silicon-bc-seeded-muon-result]] on later ckpts (stable
   step 30k→196k, resolving that memory's un-converged caveat).
6. **No-bind muon delocalises late.** Si-T muon cloud BROADENS over training
   (RMS 0.46→0.85, P<1b 1.0→0.75): no bound e⁻ to hold it. Diamond muonia stay
   localised (P<1b ≥0.94). (All runs' RMS grows modestly late = width-adaptation
   regime, [[project-muon-mcmc-width-diffusion]].)

### ★ Energy ⟂ local-spin-density DECOUPLING (→ EXP-006, striking result)
The net-spin excess (AND both ↑/↓ channels individually) plateau by ~step 50k,
but the **energy keeps falling by hundreds of mHa** for hundreds of k more steps
(the after-50k descent is ~24% of the TOTAL energy drop):
| run | ΔE after 50k plateau | excess flat at | per-spin Δ (R) |
|---|---|---|---|
| diamond-T q (#6) | −262.8 mHa (200k→end −36.2) | +0.427±0.021 | ΔN↑+0.036 ΔN↓+0.024 (1.8b) |
| diamond-BC q (#14) | −271.0 mHa (200k→end −36.6) | +0.470±0.025 | ΔN↑−0.027 ΔN↓−0.013 (2.6b) |
| silicon-BC q (#15) | −72.1 mHa (100k→end −22.5) | +0.225±0.017 | ΔN↑−0.023 ΔN↓−0.028 (3.0b) |
| **diamond-T CLASSICAL (#7)** | −284.4 mHa (200k→end −27.7) | +0.523@1.85b, ratio 6.10 | ΔN↑−0.001 ΔN↓+0.003 (1.8b) |

Both spin channels are individually FROZEN (±0.02–0.04, noise floor ±0.02–0.03) —
not cancelling drifts. **Classical #7 (fixed origin, no ZPM) rules out the
moving-origin averaging artefact** → decoupling is real. Physics: energy is
global/extensive (64-e bulk correlation, ~4 mHa/e late gain, delocalized, 2nd-order
in δψ); local spin density is intensive/local (muon+cage potential, 1st-order),
"easy" & saturates early. **Practical payoff: muon-site/muonium verdicts are valid
on partially-trained (energy-unconverged) nets from ~50k** — retro-validates
EXP-003/003b, silicon #15. Open: check r→0 CONTACT density (hyperfine) with the
fine SRPD estimator (0.1-b bins too coarse). Full writeup: experiments/EXP-006.

**Caveats:** single-checkpoint snapshots (4096 walkers) → early points noisy
(transient dips/spikes); trends robust, individual early points not. d_site
unreliable for T runs (8 equivalent T-sites, one hard-coded) — read localization
from RMS/P<1b. Verdicts on partially-trained nets but stable across late steps.

Links: [[project-experiments]] (EXP-002/003b), [[project-bc-seeded-muon-result]]
(diamond BC +0.47), [[project-diamond-results]], [[project-silicon-results]],
[[feedback-quantum-muon-srpd-no-fixed-origin]], [[reference-vmc-basin-trapping-seeding]].
Scratchpad script + figs (fig1 formation, fig2 profiles, fig3 contrast) not
committed; offer to promote to `tools/bound_state_formation.py` if wanted.
