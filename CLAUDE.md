# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **research fork** of DeepMind's [FermiNet](https://github.com/google-deepmind/ferminet)
(JAX variational Monte Carlo for many-electron wavefunctions), extended for
**muon-site / µSR studies in solids** (diamond and silicon supercells). The
upstream `ferminet/` package is largely unchanged; the research lives in a few
custom extensions and in the surrounding scaffolding (`configs/`, `jobs/`,
`tools/`, `experiments/`, `claude_memory/`).

The central scientific question is where a muon localises in the lattice (T-site
vs bond-centre) and whether it forms muonium (a bound muon+electron, i.e.
paramagnetic) — see `experiments/` and `claude_memory/`.

## Environment & commands

Use the **`ferminet-piku` conda env** for anything in this repo (running,
analysis scripts, tools):

```shell
conda activate ferminet-piku
pip install -e '.[testing]'   # install package + flake8/pylint/pytest/pytype
```

- **Run all tests:** `python -m pytest`
- **Single test file / case:** `python -m pytest ferminet/tests/networks_test.py`
  or `python -m pytest ferminet/tests/networks_test.py -k <substring>`
- **Multi-device (2 CPU) tests:** `FERMINET_CHEX_N_CPU_DEVICES=2 python -m pytest ferminet/tests/train_test.py`
- **Lint (CI gate):** `flake8 .` and `pylint --fail-under 9.5 ferminet`
- **Type check:** `pytype ferminet`

CI (`.github/workflows/ci-build.yaml`) runs lint + pylint (≥9.5) + pytype + the
full pytest suite on Python 3.11. It triggers on `main` only.

Note: the `main` branch here is upstream-tracking; day-to-day work happens on
research branches (currently `af`).

## Two ways to run FermiNet — and why configs are run as scripts

1. **Upstream CLI (small molecules/atoms):**
   ```shell
   ferminet --config ferminet/configs/atom.py --config.system.atom Li --config.batch_size 256
   ```
   Config settings are overridden with `--config.<path>` flags (ml_collections).
   All available knobs are defined in `ferminet/base_config.py`.

2. **Custom PBC/muon configs are executed directly as Python scripts**, not via
   the `ferminet` CLI, e.g. `python ferminet/configs/silicon/bc_seeded.py`.
   These configs have a `__main__` block that calls
   `jax.distributed.initialize(...)` for multi-host SLURM before importing
   anything JAX-related, then sets training hyperparameters and calls
   `train.train(cfg)`. **Import order matters**: JAX imports must stay inside
   `get_config()` / after `jax.distributed.initialize`, never at module top.
   The matching SLURM launchers live in `ferminet/jobs/<system>/*.sh` (srun
   across nodes, one config script per job).

**Inference** = re-run the same config with the optimizer disabled
(`cfg.optim.optimizer = 'none'` / `--config.optim.optimizer 'none'`) and the
same `save_path` (or `restore_path` pointing at it). The custom configs come in
training/`inference*` pairs (e.g. `t_relaxed.py` + `inference_t_relaxed.py`);
keep the two in sync (geometry, particle counts, `save_path`).

## The muon/particle extension (the main fork-specific code)

Upstream FermiNet models electrons only. This fork generalises to arbitrary
particle species via three parallel tuples in `cfg.system` (see
`base_config.py`):

- `cfg.system.particles` — count per species, e.g. `(33, 32, 1)` = up-electrons,
  down-electrons, one muon.
- `cfg.system.charges` — charge per species, e.g. `(-1., -1., 1.)`.
- `cfg.system.masses` — mass per species in electron masses, e.g.
  `(1., 1., 206.7682827)` for the muon (`MUON_MASS`).

The muon is the **last particle**. Its MCMC walkers can be seeded near a chosen
site with `cfg.mcmc.muon_init_coord` (cartesian bohr, molecule frame) and
`cfg.mcmc.muon_init_width` (Gaussian width, bohr). When `muon_init_coord` is
`None`, behaviour is identical to stock FermiNet. Seeding is how experiments test
whether a site is a true minimum vs a sampling trap (e.g. EXP-002).

Solid-state runs use periodic boundary conditions (`ferminet/pbc/`,
`cfg.system.pbc.*`, lattice vectors) plus pseudopotentials
(`cfg.system.use_pp`, `ferminet/pseudopotential.py`) and disable pretraining
(`cfg.pretrain.method = None`).

## Upstream architecture (big picture)

`train.train(cfg)` in `ferminet/train.py` is the orchestrator. It wires together:

- **Network / ansatz** — `networks.py` (FermiNet) and `psiformer.py` (attention
  ansatz), built from `network_blocks.py`; multiplicative `envelopes.py` and
  optional `jastrows.py`. `init.py` builds electron/particle initial positions.
- **Hamiltonian & local energy** — `hamiltonian.py` (kinetic via Laplacian —
  `folx` forward-Laplacian or default; potential energy), `pseudopotential.py`.
- **Loss / optimiser** — `loss.py` (VMC energy, plus excited-state objectives),
  optimised with KFAC (`curvature_tags_and_blocks.py`) or optax.
- **Sampling** — `mcmc.py` (Metropolis walker updates; this is where muon
  seeding lives), `pretrain.py` (Hartree-Fock pretraining via PySCF).
- **Observables** — `observables.py`, `density.py`, `observable/apmd.py`, spin
  (S²), dipole, density matrices; toggled via `cfg.observables.*`.
- **I/O** — `checkpoint.py` (checkpoints + walker state), `utils/writers.py`
  (`train_stats.csv`). `constants.py` holds pmap/sharding helpers.

Excited states: set `cfg.system.states = k`; default is NES-VMC, or set
`cfg.optim.objective = 'vmc_overlap'` for the ensemble penalty method.

## Research scaffolding (custom, not upstream)

- `ferminet/configs/{diamond,silicon,dft,excited}/` and top-level configs — one
  file per physical system/geometry (e.g. `bc_relaxed`, `t_relaxed`, `bc_seeded`,
  `*_classical` = muon fixed, `inference_*`). Geometries are DFT-relaxed lattice
  coordinates in units of the lattice constant `a`.
- `ferminet/jobs/<system>/*.sh` — SLURM launchers for the configs above.
- `tools/` — standalone post-processing scripts run against a run's
  `save_path`: `energy_convergence.py` / `convergence_check.py` (block-averaged
  VMC energy + drift test on `train_stats.csv`), `muon_site_analysis.py` (muon
  localisation + SRPD/spin-resolved contact density for muonium),
  `srpd_extended_radius.py`, `broaden.py`, `doppler.py`, `rhor.py`,
  `print_checkpoint.py`, `learning_curve.py`.
- `ferminet/musr/analysis/` — µSR-specific analysis (e.g. coordinate builders).
- `experiments/EXP-NNN_*.md` — the lab notebook. **Read `experiments/README.md`
  first**: every experiment follows a fixed hypothesis/H0/verdict template and is
  indexed with its verdict. Add new experiments in the same format.
- `claude_memory/` — in-repo research state (job status, per-run results,
  energy tables). `claude_memory/MEMORY.md` is the index; skim it to catch up on
  where the muon-site investigation stands and which runs are active. This is
  distinct from Claude Code's own `.claude/.../memory/` directory.

## Conventions

- Two-space indentation (upstream Google style); `.flake8` max line length 80,
  `pylintrc` is the upstream config. Match the surrounding file.
- Units are **bohr / hartree / electron-mass** throughout unless a config
  explicitly converts from angstrom.
- When editing a training config, update its inference counterpart to match.
