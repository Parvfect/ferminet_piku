---
name: project-convergence-snapshot
description: "Energy-convergence status of all 7 production training runs (from tools/energy_convergence.py) — dated snapshot, re-run to refresh"
metadata: 
  node_type: memory
  type: project
  originSessionId: 40812d63-6ed1-449d-bf5d-6c792162cfd4
---

## Energy convergence of the 7 production runs — snapshot 2026-06-17

Ran [[tool-energy-convergence]] (default tol 1e-3 E_h, 1000-step blocks, 20%
burn-in, 20-block trailing window) on every [[job-save-paths]] save_path.
Energies are the trailing-window plateau mean ± 1 SEM. **This is a dated
snapshot — energies/verdicts move as runs continue; re-run to refresh.**

| Run | Step | Plateau E (E_h) | SEM | Slope flat? | Verdict |
|---|---|---|---|---|---|
| diamond bc_relaxed pp q | 312k | -90.5988 | 0.0002 | yes | plateaued |
| diamond bc_relaxed nopp q | 812k | -609.187 | 0.0024 | yes | plateaued (noisy, all-electron) |
| diamond unrelaxed pp q | 312k | -90.6480 | 0.0003 | no | drifting down, slow (-1.3e-7/step) |
| diamond bc_relaxed classical (bc) | 268k | -90.7223 | 0.0003 | no | drifting down, slow (-1.4e-7) |
| silicon classical t-site | 144k | -62.9030 | 0.0003 | no | drifting down (-1.9e-7) |
| silicon q unrelaxed | 144k | -62.8469 | 0.0009 | no | drifting down, faster (-6.4e-7) |
| diamond classical t-site pp | 88k | -90.5335 | 0.0050 | no | strongly drifting (-3.8e-6), far off |

**Takeaways:**
- Only the two **bc_relaxed quantum-muon** runs have plateaued (slope flat
  within noise). The nopp one has a large ±2.4 mE_h bar / big block scatter
  (96-electron all-electron — inherently noisier) but is statistically flat.
- The two **youngest** runs are least converged: `diamond classical t-site
  pp` (88k steps, ~72 mE_h drift over the window — nowhere near plateau) and
  `silicon q unrelaxed` (~12 mE_h drift). They need substantially more steps.
- The other three drift only slowly (~2-4 mE_h over 19k steps), near plateau.
- diamond unrelaxed pp q (-90.6480) sits a real ~49 mE_h *below* bc_relaxed pp
  q (-90.5988) and is still slowly descending — but different geometries =
  different Hamiltonians, so this is NOT a variational-quality comparison.
- silicon q unrelaxed's permanent nan *variance* ([[potential-bugs]]) does not
  affect this check (it reads the energy column, which is finite).
