# Muon-site ground-state energy comparison

*Block-averaged plateau energies (±1 SEM) from each run's `train_stats.csv`,
refreshed 2026-07-08 via `tools/energy_convergence.py`. Interactive chart:
`muon_gse_comparison.html`; regenerate both with `gen_muon_gse_chart.py`.*

**Read:** among *quantum* runs, seeding reaches the lowest energy in both systems.
Diamond **#14 bc_seeded** (converged, −90.696 E_h) sits **26.9 mHa below** the
unseeded off-T **#8**; silicon **#17 t_seeded** / **#15 bc_seeded** sit ~17 mHa below
the unseeded runs. Classical rows fix the muon as a point charge, so they carry **no
zero-point energy** and sit artificially low — a fixed-site reference, not a
comparison. ★ marks a run that has **not** strictly converged.

## Diamond
*2×2 supercell · 16 C + μ · doublet · pseudopotential · a = 6.74 bohr. Lower E = better.*

| Run | Config | Muon site | Treatment | E (E_h) | ±SEM | Steps | Converged? |
|-----|--------|-----------|-----------|--------:|-----:|------:|------------|
| `#4` | `classical bc` | fixed BC | classical (fixed μ) | **-90.73010** | 0.00029 | 338k | ✓ converged |
| `#7` | `classical T_site` | fixed T | classical (fixed μ) | **-90.69624** | 0.00015 | 472k | ✓ converged |
| `#14` | `bc_seeded` | BC | quantum · seeded | **-90.69597** | 0.00019 | 580k | ✓ converged |
| `#13` | `classical t_relaxed` | fixed T* | classical (fixed μ) | **-90.69222** | 0.00027 | 428k | ★ descending |
| `#8` | `pp_relax_2` | off-T | quantum · unseeded | **-90.66904** | 0.00029 | 280.2k | ★ frozen (final) |
| `#6` | `unrelaxed pp` | T | quantum · unseeded | **-90.66312** | 0.00017 | 522k | ✓ converged |
| `#12` | `t_relaxed` | T | quantum · unseeded | **-90.65979** | 0.00023 | 478k | ★ noise-limited |
| `#5` | `bc (original)` | off-T | quantum · unseeded | **-90.59794** | 0.00151 | 336k | ★ noise-limited |
| `#18` | `t_seeded` | T | quantum · seeded | **-90.54469** | 0.00208 | 120k | ★ warm-up |

## Silicon
*2×2×2 supercell · 16 Si + μ · doublet · pseudopotential · a = 10.26 bohr. Lower E = better.*

| Run | Config | Muon site | Treatment | E (E_h) | ±SEM | Steps | Converged? |
|-----|--------|-----------|-----------|--------:|-----:|------:|------------|
| `#11` | `classical t_relaxed` | fixed T* | classical (fixed μ) | **-62.92218** | 0.00020 | 306.6k | ★ diverged @306k |
| `#2` | `classical (T)` | fixed T | classical (fixed μ) | **-62.91337** | 0.00015 | 223.7k | ★ frozen (final) |
| `#15` | `bc_seeded` | BC | quantum · seeded | **-62.89170** | 0.00022 | 196k | ★ descending |
| `#17` | `t_seeded` | T (relaxed) | quantum · seeded | **-62.89124** | 0.00051 | 224k | ★ noise-limited |
| `#10` | `t_relaxed` | T (fled) | quantum · unseeded | **-62.87392** | 0.00034 | 138k | ★ frozen (final) |
| `#16` | `classical bc_relaxed` | fixed BC | classical (fixed μ) | **-62.86852** | 0.00229 | 61.4k | ★ warm-up |
| `#1` | `unrelaxed` | T | quantum · unseeded | **-62.86151** | 0.00047 | 180k | ★ frozen (final) |
| `#9` | `bc_relaxed` | T (fled) | quantum · unseeded | **-62.86146** | 0.00076 | 138k | ★ frozen (final) |

## Notes & caveats
- **Classical ≠ quantum.** Classical runs omit muon zero-point energy and sit lower by
  construction — a different quantity, shown only as fixed-site references.
- **Diamond and silicon are separate scales** (~−90 vs ~−62 E_h) and are never compared
  across the two tables.
- **★ = not converged** by the block-drift test (`tools/energy_convergence.py`, tol 1 mHa).
  *noise-limited* = slope statistically flat but drift just over tol (effectively
  plateaued); *frozen* = training stopped; *descending / warm-up* = still actively falling.
- **Only 4 runs are strictly converged:** diamond #4, #7, #6, #14. All silicon runs are
  still descending or noise-limited — treat their energies as upper bounds.
- **Silicon #11** diverged at step 306.6k; the value shown is its last clean block.
  **Diamond #3** (all-electron, no pseudopotential, −609 E_h) is a different Hamiltonian
  and is omitted.
