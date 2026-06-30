---
name: qe-bc-relaxation-diamond
description: "QE DFT relaxation of the muon (H) at the diamond bond-center site, in ~/qe/diamond_test; how the unpaired-electron (muonium doublet) spin-polarized version was set up"
metadata:
  node_type: memory
  type: project
---

## QE relaxation of the BC-site muon in diamond

Lives in `/home/u6em/parvfect.u6em/qe/diamond_test/` (NOT under the ferminet
repo). These Quantum ESPRESSO `relax` runs DFT-relax the C lattice around a
muon (modelled as an H atom) at the **bond-center (BC) site** — the geometry
that feeds the `bc_relaxed` FermiNet runs (see [[research-goal-muon-site]]).

**Key input files (note: file is `bc_2_relax.in`, the user sometimes calls it
`bc_relax_2.in`):**
- `bc_2_relax.in` / `.out` — calculation='relax' (ionic, fixed cell), 17 atoms
  (16 C + 1 H), H at (0.446,0.446,0.446) Å. **Non-spin-polarized.**
- `bc_2.in` — calculation='vc-relax' variant, H at a different position
  (2.23,2.23,0.446).
- All use PAW PPs (C.pbe-n-kjpaw, H.pbe-kjpaw), ecutwfc=50, ecutrho=400,
  2×2×2 k-points, MP smearing degauss=0.02, conv_thr=1e-8.

## Unpaired-electron version (2026-06-18)

The original `bc_2_relax.in` has **65 valence electrons** (16 C × 4 + 1 H = 65,
odd) but runs spin-UNpolarized, so the odd electron is fractionally smeared
across spin-degenerate states — NOT a real unpaired electron. The FermiNet
runs we actually simulate use `particles=(33,32)` = net spin +1, i.e. one
unpaired up-electron (a **muonium doublet**). So the DFT relaxation should
match that.

Created `bc_2_relax_unpaired.in` (copy of `bc_2_relax.in`) adding to &SYSTEM:
```
nspin             = 2
tot_magnetization = 1            ! n_up - n_down = 1 unpaired electron
starting_magnetization(1) = 0.0  ! C
starting_magnetization(2) = 1.0  ! H (muon electron seed)
```
Electron count is unchanged (65, already odd); we only turn on collinear spin
polarization and constrain the total moment to 1. starting_magnetization is
just the SCF seed (per ATOMIC_SPECIES order: 1=C, 2=H). Original file + its
`.out` kept as the no-unpaired-electron reference.

**How to run QE here:** `module load libfabric` then
`pw.x -in <file>.in > <file>.out` from `~/qe/diamond_test/`. numpy etc. only
in the `ferminet-piku` conda env (see [[feedback-python-env]]).

## Result (2026-06-18): spin vs non-spin relax comparison

Ran `bc_2_relax_unpaired.out`. Both runs converged in 5 BFGS steps to the
same basin. **No major geometry difference** (prediction held):
- Max atomic displacement spin-vs-nonspin = **0.0062 Å** (the two muon-bonded
  C atoms), RMS 0.0033 Å over 17 atoms. Falls off with distance from muon.
- Muon (H) stayed pinned at the BC site (0.446³) in both — did NOT slide to T.
- C–muon bond: 1.0900 Å (spin) vs 1.0961 Å (non-spin) — doublet contracts the
  symmetric C–μ–C bridge by ~0.012 Å total. Sub-1%, negligible.
- Magnetization total 1.00 μB / absolute 1.08 μB → one fairly clean unpaired
  electron, only minor spin texture (delocalized level, not tight muonium).
- **Energy:** spin-constrained (M=1) is **+318 meV HIGHER** than non-spin
  smeared (−295.60317 vs −295.62655 Ry). NOT "spin is unfavorable" — the old
  run smears the odd e⁻ fractionally over both channels (unphysical
  paramagnetic avg); the new one commits to a real S_z=½ doublet. The +318 meV
  is the moment-formation cost, consistent with BC being diamagnetic-favoring
  in diamond (cf [[project-diamond-classical-bc-muon-result]]).

**Takeaway:** existing bc_relaxed geometry was already a fine approximation,
but `bc_2_relax_unpaired.in` is the faithful match to the FermiNet doublet
(33↑/32↓) and should be the structure fed to the bc_relaxed runs.

## Silicon BC relaxation (2026-06-18)

Built the silicon analogue by scaling the diamond 2×2×2 supercell by
s = a_Si/a_diamond = 5.42934/3.568 = 1.521676 (a_Si = 5.42934 Å = 10.26 bohr,
user-supplied). Same fractional structure: 16 Si + 1 H (muon) at BC =
a_Si/8·(1,1,1) = 0.6786675 Å. 16×4+1 = 65 e⁻ (odd), so tot_magnetization=1
identical to diamond. Cutoffs kept 50/400 Ry (Si PAW recommends 44/175, so
safe). Files in `~/qe/diamond_test/`:
- `si_bc_relax.in` (non-spin baseline, outdir ./tmp_si)
- `si_bc_relax_unpaired.in` (nspin=2, tot_magnetization=1, outdir ./tmp_si_unpaired)

**Pseudo:** had to download `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (none existed; only
C+H were in `~/qe/psuedo/`). Got it from
pseudopotentials.quantum-espresso.org/upf_files/ — same pslibrary 1.0.0
PAW-PBE family (ADC) as the C pseudo, Z_valence=4.

**Status:** both submitted (background), comparison pending. NB silicon
physics differs from diamond — prior SRPD found silicon q-muon DIAMAGNETIC /
no muonium ([[project-silicon-quantum-muon-result]]), and BC is the known
anomalous-muonium (Mu*) site in Si, so the relax behaviour may differ.
