---
name: diamond-classical-bc-muon-result
description: "SRPD for diamond classical muon (pp_T, H fixed at BC site) — diamagnetic, high electron density; muonium is a T-site property not BC"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23883b38-2982-49b8-9517-0fb762f321bc
---

## Diamond classical muon pp_T (BC site) SRPD — validated 2026-06-15

- Config: `ferminet/configs/diamond/bc_relaxed/inference_pp_T.py`. Classical
  muon = fixed H at BC site (0.12504759a)³ = (0.843,0.843,0.843) bohr, a=6.74,
  particles (33,32) (65 electrons, NO muon particle), mol.charge=1, PP.
  BC-relaxed C geometry (C0 & C2 pushed apart around the H).
- Positions: `/projects/u6em/parv/diamond/unpaired/classical/bc/inference/positions/`
  shape (4,128,195)=65 particles. Run via `tools/muon_site_analysis.py
  diamond_classical_bc srpd` (fixed-origin branch, origin = H/BC position).

**⚠️ "DIAMAGNETIC" VERDICT SUPERSEDED (2026-06-30).** The contact-only (rmax=1)
SRPD below sees g_up≈g_down inside 1 bohr and looked diamagnetic — but that's just
the symmetric +1 screening cloud. The extended-radius net spin
(`tools/srpd_extended_radius.py diamond_classical_bc 4.5 90 2`) shows a LOCALIZED
net-spin excess rising to **+0.53 e⁻ at the bond scale (~2.6 bohr)**, 5–6× above the
uniform-band line ⇒ **anisotropic, bond-centred MUONIUM**, not diamagnetic. The muon
is CLAMPED here (no ZPM) so it's an electronic property of BC. Matches the quantum
BC-seeded run (+0.47, [[project-bc-seeded-muon-result]]). NB these positions are from
ckpt **222000**, not the converged **338000** net (#4 plateau −90.73010) → re-run the
inference at 338k to tighten. Original (contact-only) read below:

**Result (contact-only, now incomplete): dense bonding charge.**
- contact g_up~0.22, g_down~0.26 (unpolarised inside 1 bohr — screening cloud, NOT
  the whole story; see net-spin correction above).
- electrons within 1 bohr: up=0.42, down=0.38, total ~0.79 — much higher than
  the T-site cases (~0.50), because BC sits in the C-C bonding charge.

**KEY PHYSICS — muonium in diamond is a T-SITE property, not BC:**

| diamond run | site | e⁻ <1 bohr (up/dn) | contact spin | character |
|---|---|---|---|---|
| unrelaxed quantum [[unrelaxed-pp-muon-result]] | T (0.25,0.25,0.75) | 0.38/0.12 | ~5:1 up | MUONIUM |
| bc_relaxed quantum [[bc-relaxed-pp-muon-result]] | T (0.5,0.55,0.5) | 0.25/0.27 | ~1.3:1 | weak |
| classical pp_T (this) | BC (0.125³) | 0.42/0.38 | ~1:1 (slight down) | DIAMAGNETIC |

At the open tetrahedral interstitial (low density) the muon binds a spin-up
electron → muonium; forced into the bond centre (high bonding charge) it is
diamagnetic. Silicon muon sits at T but is diamagnetic there too (narrower gap)
— see [[silicon-quantum-muon-result]].
