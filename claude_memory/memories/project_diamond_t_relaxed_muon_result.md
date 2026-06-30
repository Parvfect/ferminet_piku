---
name: project-diamond-t-relaxed-muon-result
description: "diamond T-relaxed q-muon HOLDS the expanded relaxed T-cage (centered, +0.017 bohr expansion) AND is MUONIUM — contrast with silicon which fled its contracted cage"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8c3e8cf7-77a6-4a72-9947-de89afb46168
---

Diamond **T-relaxed** quantum muon (#12, job family `muon_d_qpp_t_rel`) position
inference (job 5420036 `muon_d_qpp_t_rel_inf`) — analysed 2026-06-29. All
1000/1000 positions dumped (512k samples) from
`/projects/u6em/parv/diamond/unpaired/t_relaxed/pp/inference/positions/`.

**SITE — HOLDS the relaxed (expanded) T-cage (★ key contrast with silicon):**
muon at a genuine **T interstitial**, cubic-frac ≈ (0.50,1.00,1.00) (symmetry-
equiv to seed site). Well localized: median |r| 0.59 bohr, 93% within 1 bohr,
RMS radius 0.66 bohr (mildly anisotropic 0.47/0.33/0.33). Sits **dead-center**
in its cage (0.045 bohr off centroid), 4 C at ~2.94 bohr. Cage carbons radially
displaced **OUTWARD +0.017 bohr (~+0.6%)** — matches DFT +0.4% T-cage EXPANSION.
Scan confirms all equivalent T-cages uniformly expanded ~+0.017 bohr.
→ Prediction CONFIRMED: diamond muon STAYS in the expanded relaxed cage WITHOUT
seeding. OPPOSITE of silicon, whose q-muon FLED 8.47 bohr to an unrelaxed cage
(Si cage CONTRACTED −1.1%). Expansion holds, contraction expels. See
[[project-silicon-quantum-muon-result]] and EXP-003 caveat.

**SRPD — MUONIUM (paramagnetic):** strong spin-up contact density, g_up≈6:1 over
g_down at contact (g_up/bulk 7.2 vs g_dn/bulk 0.86 at r≈0.03). Extended-radius
net spin: localized_excess rises to ~+0.4 and PLATEAUS within lattice scale
(2–3.6 bohr) before WS-edge artifact → localized bound electron = muonium
(partial/loose, diamond's reduced-contact "anomalous" muonium).
→ Contrasts silicon (diamagnetic, localized_excess ≈0). All 3 diamond runs
(unrelaxed, bc_seeded, t_relaxed) → T-site MUONIUM; diamond forms muonium,
silicon stays diamagnetic.

CAVEAT: #12 not fully converged (−90.636 @248k, slope −2.1e-7) at restored ckpt
→ partially-trained net, but site/spin signatures unambiguous.

Tools: `tools/muon_site_analysis.py diamond_t_relaxed {site|spread|srpd}` +
`tools/srpd_extended_radius.py diamond_t_relaxed 6.0 120 4`.
