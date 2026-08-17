---
name: feedback-quantum-muon-srpd-no-fixed-origin
description: "For a QUANTUM (moving) muon, SRPD inference must use use_fixed_origin=False; fixed origin is ONLY for classical fixed-muon runs"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b31323f6-35f2-461d-ad0c-521f9e20227b
---

When configuring an in-run SRPD (`cfg.observables.srpd`) inference for a
**quantum muon** run (particles `(nup, ndown, 1)`, e.g. bc_seeded, t_seeded,
t_relaxed quantum), set **`cfg.observables.srpd.use_fixed_origin = False`** (the
default) and do NOT set `origin_coord`.

**Why:** In `ferminet/observables.py` the SRPD estimator branches on
`use_fixed_origin`:
- `True` → literally commented `# No muon`; it unpacks `nspins` as a **2-tuple**
  (`n_up, n_down = nspins`) and measures electron density from the fixed cartesian
  `origin_coord`. This is only correct for the **classical** fixed-muon runs whose
  walkers have no muon particle. Applied to a `(33,32,1)` quantum walker it is both
  the wrong physics AND structurally broken (3-tuple unpack).
- `False` → unpacks the 3-tuple (`n_up, n_down, _ = nspins`) and measures density
  relative to the **muon's instantaneous position** (`rvec = x[:-1] - x[-1]`,
  last particle = muon). This is what a delocalised quantum muon needs.

**How to apply:** Quantum-muon SRPD inference = `srpd.calculate=True`,
`use_fixed_origin=False`, no `origin_coord`. Match `configs/silicon/inference.py`.
Fixed origin (`use_fixed_origin=True` + `origin_coord=np.array([site*a]*3)`) is
reserved for the classical runs (`classical_muon/t_site_inference_pp.py`,
`silicon/t_site_inference.py`). The muon's actual localisation/spin structure at a
site is instead recovered from dumped positions via
`tools/srpd_extended_radius.py` ([[reference-tools]]). User corrected me on this
2026-07-06 while setting up the diamond bc_seeded position+SRPD inference.
Related: [[project-bc-seeded-muon-result]], [[training-monitoring]].
