---
name: research-goal-muon-site
description: "Standing research goal: verify FermiNet optimization places the muon at the expected lattice site (BC for relaxed geometries, T or BC for unrelaxed)"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd7babd3-df02-4561-aef6-eb933702fa94
---

## Current goal: muon site verification via SRPD

**Goal:** Confirm that the variational optimization (quantum muon as a
third particle species) converges to the physically expected muon
localization site:

- **BC-relaxed models** (C atoms DFT-relaxed around the bond-center site —
  `bc_relaxed/pp.py`, `bc_relaxed/nopp.py`, and their inference configs):
  the muon should localize at the **BC site**.
- **Unrelaxed models** (ideal diamond positions — `diamond/pp.py`
  (unrelaxed) and silicon unrelaxed): the muon should localize at the
  **T site or the BC site** (either is consistent with expectation).

**Why:** This is the core physics question behind the unpaired-electron
quantum-muon runs — does the trained wavefunction actually represent the
muon sitting where it's physically expected to, or does it drift to a
different interstitial site?

**How to apply:** The SRPD (spatial radial probability density) observable,
just turned on for the 4 quantum-muon inference runs in
[[current-status]] (jobs 5247625, 5247627, 5247628, 5247629), is the tool
for this — it gives the muon's spatial distribution relative to lattice
sites. When these inference runs produce SRPD output, analyze the muon
density peak location and classify it as BC-site, T-site, or neither, for
each of: BC-relaxed PP, BC-relaxed no-PP, unrelaxed diamond, unrelaxed
silicon.

**Existing relevant data point:** [[bc-relaxed-pp-new-muon-result]] found
the muon localizing at the **T-site** (not BC) in a BC-relaxed PP run
(`inference_pp_new.py`, paired-electron, 2026-06-05) — i.e. a mismatch with
the BC-relaxed expectation above. That was flagged as possibly a
training/relaxation-mismatch issue. The new SRPD runs (unpaired-electron,
job 5247625 for BC-relaxed PP) will give a fresh, directly comparable
result to check whether this still holds.
