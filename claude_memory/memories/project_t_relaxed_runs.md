---
name: project-t-relaxed-runs
description: "T-relaxed (DFT-relaxed-around-T-site) muon runs: silicon pair SUBMITTED 2026-06-22; diamond QUANTUM SUBMITTED 2026-06-23 (job 5358557); diamond classical not yet submitted"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6b661f9c-06e3-459d-abb5-5c6dec7d5115
---

## T-relaxed muon runs (added 2026-06-22, branch `af`)

New geometry variant: host atoms DFT-relaxed around a muon sitting at the
tetrahedral (T) interstitial, i.e. the relaxed analogue of the unrelaxed T-site
runs. Four configs were added via git pull (commit `2a3d5a8` "Adding t relaxed").
Each is a neutral open-shell doublet: 65 e⁻ (33↑/32↓, spin=1), PP on host atoms.
Quantum = sampled muon (q=+1, m=206.77, particles `(33,32,1)`, `mol.charge=0`, no
H atom). Classical = fixed-H nucleus at the relaxed T-site (particles `(33,32)`,
`mol.charge=1`). All verified: charge balances to NET 0; classical fixed-H sits at
the quantum ideal T-site; nearest-4 host distances Si ≈4.394 bohr, C ≈2.930 bohr.

Relaxation is small (per config comments): diamond max C displacement 0.0414 bohr
(+0.4% T-cage expansion); silicon only the 4 coordinating Si move ~0.0487 bohr
inward (−1.1% T-cage contraction). Coordinate sources:
`musr/analysis/diamond_coordinates.py` (`atomic_positions_t_relaxed_cubic`) and
`musr/analysis/silicon_coordiantes.py` (`atomic_positions_t_relaxed`).

| # | System | Type | Config | Job script | save_path | Status |
|---|---|---|---|---|---|---|
| 10 | Silicon | quantum | `configs/silicon/t_relaxed.py` | `jobs/silicon/t_relaxed.sh` | `/projects/u6em/parv/silicon_unpaired/t_relaxed` | **SUBMITTED job 5334500** (q, `muon_silicon_q_t_relaxed`) |
| 11 | Silicon | classical | `configs/silicon/t_relaxed_classical.py` | `jobs/silicon/t_relaxed_classical.sh` | `/projects/u6em/parv/silicon_unpaired/classical/t_relaxed` | **SUBMITTED job 5334501** (`muon_silicon_c_t_relaxed`) |
| 12 | Diamond | quantum | `configs/diamond/t_relaxed/pp.py` | `jobs/diamond_2x2/t_relaxed/pp.sh` | `/projects/u6em/parv/diamond/unpaired/t_relaxed/pp` | **SUBMITTED job 5358557** (2026-06-23, `muon_d_qpp_t_rel`, PD) |
| 13 | Diamond | classical | `configs/diamond/t_relaxed/classical.py` | `jobs/diamond_2x2/t_relaxed/classical.sh` | `/projects/u6em/parv/diamond/unpaired/classical/t_relaxed/pp` | **RUN NOT STARTED** — not yet submitted |

**Submitted (2026-06-22):** the two silicon jobs (PENDING at submit). They are
now in the active daily set ([[training-monitoring]], #10/#11). Submit-from dir
`ferminet/jobs/silicon/`; relative `--output` → `t_relaxed.out` /
`t_relaxed_classical.out` in that dir.

**Submitted (2026-06-23):** diamond quantum #12 (`sbatch pp.sh` from
`ferminet/jobs/diamond_2x2/t_relaxed/`, job 5358557, PENDING; output `pp.out`
in that dir; save dir was empty). Diamond classical #13 NOT yet submitted — to
launch: `sbatch classical.sh` from the same dir (output `classical.out`).

All 4 save dirs created empty, no collisions. NOTE the empty leftover capital
`T_relaxed` dirs (`silicon_unpaired/T_relaxed`, `diamond/unpaired/T_relaxed`, both
Jun 19) are unrelated — new configs use lowercase `t_relaxed`.

Goal context: completes the relaxed-vs-unrelaxed T-site comparison for the
[[research-goal-muon-site]] muon-localisation question. See [[job-save-paths]].
