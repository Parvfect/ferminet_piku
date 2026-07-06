# Experiments

Lab notebook for muon-site research. One file per experiment, named
`EXP-NNN_short-slug.md`. Each entry follows the template below so results stay
comparable and decisions are traceable.

## Index

| # | Title | Verdict | Date |
|---|-------|---------|------|
| [001](EXP-001_muon_offT_vs_bc_classical.md) | Is the FermiNet off-T muon the true site, or a sampling trap? | **H0 rejected** — BC is the true minimum (1.90 eV below off-T); off-T is a trap. *Why* the optimiser selects off-T is unresolved. | 2026-06-24 |
| [002](EXP-002_bc_seeded_muon_run.md) | BC-seeded quantum run — does the net hold BC or drain to off-T? | planned (awaiting verification) | 2026-06-24 |
| [003](EXP-003_silicon_t_relaxed_bound_state.md) | Does the silicon T-relaxed quantum muon form a bound state (muonium)? | running (inference 5415683) | 2026-06-29 |
| [004](EXP-004_muon_width_burnin_diffusion.md) | Does a wider muon proposal + longer burn-in un-trap the muon (off-T → BC) without seeding? | implemented, ready to run (H0 the likely prior — run as diagnostic) | 2026-07-06 |

## Template

```markdown
# EXP-NNN: <title>

- **Date:** <YYYY-MM-DD>   **Status:** <planned | running | done>
- **System:** <e.g. diamond 2x2, BC-relaxed cage, pp_relax_2>

## Question / Motivation (why)
<What prompted this? What decision does the answer inform?>

## Hypothesis (H1)
<The effect we expect, stated so it could be falsified.>

## Null hypothesis (H0)
<The skeptical default: "no effect" / the simplest explanation. This is what we
try to reject.>

## Alternative explanations
<Other things that could produce the same observation (confounds), and how the
design controls for each.>

## Method (experiment detail)
<Exact setup: code/inputs, parameters, what is varied vs held fixed, files.>

## Result
<Numbers, tables, plots. Raw enough to re-check.>

## Verdict
<H0 rejected / not rejected, and the physical conclusion. State confidence and
caveats.>

## Next / current steps
<Follow-ups, open questions, what this unblocks.>
```
