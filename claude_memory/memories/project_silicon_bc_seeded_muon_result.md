---
name: project-silicon-bc-seeded-muon-result
description: "★ Silicon BC-seeded (#15) q-muon HOLDS BC + weak partial muonium (~+0.25 e⁻); silicon muonium is SITE-DEPENDENT (diamagnetic at T, weak at BC)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9d7dbc59-9b79-426b-aa46-3fc2a07715a2
---

## Silicon BC-seeded muon (#15, EXP-002 silicon analogue) — 2026-07-01 ★

Position inference (job 5451338) off #15 ckpt **78000 (NOT converged** — post
blow-up/recovery, ~30 mHa above proper Si GSE). 1000 positions, 512k muon samples.

**★ KEY RESULT: BC-seeding PINS the silicon muon at BC (unlike unseeded #9 which
fled to a T cage), and — unlike every diamagnetic silicon T-site result — it
carries a WEAK/PARTIAL localized net-spin excess. Silicon muonium is
SITE-DEPENDENT: diamagnetic at T, weak bond-centred muonium at BC.**

- **Site — HOLDS BC.** circ-mean 0.18 bohr from ideal BC (cubic-frac
  [0.139,0.116,0.130]); 2 nearest Si {8,0} at 2.94/3.06 bohr (≈half-bond 2.99),
  2.47 bohr gap → bond centre. Tight: RMS 0.756 bohr, 84.5% within 1 bohr.
  Contrast [[project-silicon-bc-relaxed-muon-result]] (#9 unseeded fled to T cage).
- **Net spin (extended-radius, WS 7.25 bohr):** `localized_excess` rises to
  **~+0.26 e⁻ @ ~3.3–3.6 bohr** (~5× uniform line @3 bohr), then DECAYS (+0.16
  @5.7) — genuine but weak, ~half the diamond BC ~+0.47, never reaches +1.
  → weak/partial **anisotropic BC muonium**, not a strong bound state.

**Caveat:** #15 not converged — re-run inference at a later ckpt to confirm the
+0.25 is stable.

## Classical T-relaxed cross-check (#11) — 2026-07-01
Inference (job 5451339) off #11 ckpt **196000 (well-trained)**, fixed-H at relaxed
T-site, 65 e⁻ (33,32). Extended-radius net-spin `localized_excess` **≈0/slightly
NEGATIVE** within lattice scale (−0.03→−0.06), no up-contact enhancement (g↑ 8.79 <
g↓ 11.58 at contact). **DIAMAGNETIC — no bound state**, even for the ZPM-free fixed
muon (contact upper bound). Confirms EXP-003 H0 on a converged net (the quantum #10
verdict rested on an un-converged net). See [[research-goal-muon-site]].

Compare diamond: [[project-bc-seeded-muon-result]] (diamond BC muonium ~+0.47),
[[project-diamond-classical-bc-muon-result]]. Analysis: `tools/srpd_extended_radius.py
silicon_bc_seeded 6.0 120 2` / `... silicon_t_relaxed_classical 6.0 120 2`. Cases +
inference configs/scripts added 2026-07-01. Documented in experiments/EXP-002 +
EXP-003.
