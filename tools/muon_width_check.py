#!/usr/bin/env python3
"""Inspect the per-species MCMC proposal width for the muon from checkpoints.

Motivation
----------
In these runs the muon is declared as the *last* particle species
(`cfg.system.particles = (n_up, n_down, 1)`) and is ~207x heavier than an
electron. With `cfg.mcmc.sample_all = False` (the default, used by the
diamond/silicon `pp` configs), the MCMC move width is a per-species array
(`mcmc.py:make_mcmc_step` / `update_mcmc_width`). All species start at the same
`cfg.mcmc.move_width` (default 0.02) and then adapt *independently* to hold each
species' acceptance in [0.50, 0.55].

Because the muon's true distribution is sharply peaked (heavy mass -> 1/m
kinetic term in `hamiltonian.py`), its adapted width is driven DOWN, well below
the electron width. A collapsed muon width means the muon diffuses very slowly
between interstitial basins (T / BC / off-T) and can stay trapped at whatever
site it was initialised at. This tool reads the saved `mcmc_width` (and recent
per-species acceptance `pmoves`) out of checkpoints so you can see, and track,
that collapse.

The width is stored in each `qmcjax_ckpt_<step>.npz` (device-0 slice) by
`checkpoint.save`; no JAX is needed to read it.

Usage
-----
    # single checkpoint
    python tools/muon_width_check.py /path/run/qmcjax_ckpt_500000.npz

    # a run directory (save_path): tabulate the width evolution over all ckpts
    python tools/muon_width_check.py /path/run [--particles 33,32,1]

    # only the most recent checkpoint in a directory
    python tools/muon_width_check.py /path/run --last

Options
-------
    --particles a,b,c   Species counts, to label rows (e.g. up-e,down-e,muon)
                        and confirm the last species is the single muon.
    --last              With a directory, report only the latest checkpoint.
    --max-rows N        Cap the number of checkpoints tabulated (default 40;
                        evenly subsamples if more are present).
"""
import os
import sys
import argparse
import numpy as np

# Naive free-particle reference: a proposal width tuned to fixed acceptance in a
# quadratic well scales ~ 1/sqrt(m). This is NOT a target, just a rough sanity
# scale for "how much smaller should the muon width plausibly be".
MUON_MASS = 206.7682827


def extract_step(filename):
    """Step number from 'qmcjax_ckpt_XXXXXX.npz'; -1 if unparseable."""
    try:
        return int(os.path.basename(filename).split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        return -1


def find_checkpoints(path):
    """Return list of (step, filepath) for a ckpt file or a run directory."""
    if os.path.isfile(path):
        return [(extract_step(path), path)]
    if os.path.isdir(path):
        ckpts = [
            (extract_step(f), os.path.join(path, f))
            for f in os.listdir(path)
            if 'qmcjax_ckpt_' in f and f.endswith('.npz')
        ]
        ckpts = [c for c in ckpts if c[0] >= 0]
        return sorted(ckpts, key=lambda c: c[0])
    raise FileNotFoundError(f'no checkpoint file or directory at {path}')


def read_width(filepath):
    """Return (step, width_array, pmoves_or_None) from one checkpoint.

    width_array is 1-D of length nspecies (a length-1 array for the scalar /
    sample_all=True case). pmoves is (nspecies, adapt_frequency) or None.
    """
    with open(filepath, 'rb') as f:
        ckpt = np.load(f, allow_pickle=True)
        width = np.atleast_1d(np.asarray(ckpt['mcmc_width'], dtype=float))
        t = int(np.asarray(ckpt['t']).tolist())
        pmoves = ckpt['pmoves'] if 'pmoves' in ckpt.files else None
        if pmoves is not None:
            pmoves = np.asarray(pmoves, dtype=float)
    return t, width, pmoves


def recent_acceptance(pmoves, nspecies):
    """Mean acceptance per species over the filled (nonzero) buffer entries.

    pmoves is the rolling buffer update_mcmc_width writes into; unfilled slots
    are 0. We average over nonzero entries so a partially-filled buffer still
    gives a sensible recent rate. Returns array (nspecies,) or None.
    """
    if pmoves is None:
        return None
    p = np.atleast_2d(pmoves)
    if p.shape[0] != nspecies:
        # scalar (sample_all=True) buffer is 1-D; not per-species.
        return None
    out = np.full(nspecies, np.nan)
    for k in range(nspecies):
        row = p[k]
        nz = row[row != 0.0]
        if nz.size:
            out[k] = nz.mean()
    return out


def species_labels(nspecies, particles):
    """Human labels for each species row; last species assumed to be the muon."""
    if particles is not None and len(particles) == nspecies:
        labels = []
        for k, n in enumerate(particles):
            if k == nspecies - 1 and n == 1:
                labels.append(f'muon      (n={n})')
            elif k == 0:
                labels.append(f'e-up      (n={n})')
            elif k == 1:
                labels.append(f'e-down    (n={n})')
            else:
                labels.append(f'species{k} (n={n})')
        return labels
    base = [f'species{k}' for k in range(nspecies)]
    if nspecies >= 1:
        base[-1] += ' (muon?)'
    return base


def report_single(step, width, pmoves, particles):
    """Detailed one-checkpoint report."""
    nspecies = width.shape[0]
    if nspecies == 1:
        print(f'  step {step}: single shared width = {width[0]:.5g}')
        print('  NOTE: scalar width => cfg.mcmc.sample_all=True (or an old '
              'scalar-width ckpt).\n        No per-species / muon-specific width '
              'is stored for this run.')
        return

    labels = species_labels(nspecies, particles)
    acc = recent_acceptance(pmoves, nspecies)
    muon_w = width[-1]
    elec_w = width[:-1]

    print(f'  step {step}:  {nspecies} species')
    hdr = f'    {"species":<18}{"width":>12}{"recent_acc":>13}'
    print(hdr)
    print('    ' + '-' * (len(hdr) - 4))
    for k in range(nspecies):
        acc_s = f'{acc[k]*100:>11.1f}%' if acc is not None and not np.isnan(
            acc[k]) else f'{"n/a":>12}'
        print(f'    {labels[k]:<18}{width[k]:>12.5g}{acc_s}')

    print()
    if elec_w.size:
        mean_elec = float(np.mean(elec_w))
        ratio = muon_w / mean_elec if mean_elec else float('nan')
        print(f'    muon width / mean electron width = {ratio:.3g}')
        print(f'    (naive 1/sqrt(m_mu) reference     = '
              f'{1.0/np.sqrt(MUON_MASS):.3g}  -- heuristic only)')
        if ratio < 0.25:
            print('    -> muon width is heavily collapsed vs electrons: slow '
                  'muon mixing;\n       high risk the muon is pinned to its '
                  'initialisation site.')
        elif ratio < 0.6:
            print('    -> muon width notably smaller than electrons '
                  '(expected for a heavy particle).')
        else:
            print('    -> muon width comparable to electrons.')


def report_evolution(ckpts, particles, max_rows):
    """Tabulate width evolution across many checkpoints."""
    if len(ckpts) > max_rows:
        idx = np.linspace(0, len(ckpts) - 1, max_rows).round().astype(int)
        idx = sorted(set(idx.tolist()))
        ckpts = [ckpts[i] for i in idx]

    rows = []
    nspecies = None
    for step, path in ckpts:
        try:
            t, width, pmoves = read_width(path)
        except Exception as e:  # pylint: disable=broad-except
            print(f'  ! skip {os.path.basename(path)}: {e}')
            continue
        nspecies = width.shape[0]
        rows.append((t, width))

    if not rows:
        print('  no readable checkpoints.')
        return

    if nspecies == 1:
        print('  scalar width (sample_all=True): no per-species muon width.\n')
        print(f'    {"step":>10}{"width":>12}')
        for t, width in rows:
            print(f'    {t:>10}{width[0]:>12.5g}')
        return

    labels = species_labels(nspecies, particles)
    print(f'  species (last = muon): {", ".join(l.split()[0] for l in labels)}\n')
    colw = 11
    hdr = f'    {"step":>10}' + ''.join(
        f'{l.split()[0]:>{colw}}' for l in labels) + f'{"muon/e":>10}'
    print(hdr)
    print('    ' + '-' * (len(hdr) - 4))
    for t, width in rows:
        mean_elec = float(np.mean(width[:-1])) if nspecies > 1 else float('nan')
        ratio = width[-1] / mean_elec if mean_elec else float('nan')
        cells = ''.join(f'{w:>{colw}.4g}' for w in width)
        print(f'    {t:>10}{cells}{ratio:>10.3g}')
    print('\n  muon/e = muon width / mean electron width. A falling ratio over '
          'training\n  is the signature of the muon proposal collapsing '
          '(slow site-to-site mixing).')


def parse_particles(s):
    if not s:
        return None
    try:
        return [int(x) for x in s.split(',') if x.strip() != '']
    except ValueError:
        raise argparse.ArgumentTypeError(
            'particles must be comma-separated ints, e.g. 33,32,1')


def main():
    ap = argparse.ArgumentParser(
        description='Check the per-species / muon MCMC proposal width stored in '
                    'FermiNet checkpoints.')
    ap.add_argument('path', help='checkpoint .npz file or a run directory')
    ap.add_argument('--particles', type=parse_particles, default=None,
                    help='species counts to label rows, e.g. 33,32,1')
    ap.add_argument('--last', action='store_true',
                    help='with a directory, report only the latest checkpoint')
    ap.add_argument('--max-rows', type=int, default=40,
                    help='max checkpoints to tabulate (default 40)')
    args = ap.parse_args()

    try:
        ckpts = find_checkpoints(args.path)
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    if not ckpts:
        print(f'Error: no qmcjax_ckpt_*.npz found under {args.path}',
              file=sys.stderr)
        sys.exit(1)

    single = os.path.isfile(args.path) or args.last or len(ckpts) == 1
    if single:
        step, path = ckpts[-1]
        print(f'Checkpoint: {path}')
        try:
            t, width, pmoves = read_width(path)
        except Exception as e:  # pylint: disable=broad-except
            print(f'Error reading checkpoint: {e}', file=sys.stderr)
            sys.exit(1)
        report_single(t, width, pmoves, args.particles)
    else:
        print(f'Run directory: {args.path}  ({len(ckpts)} checkpoints)\n')
        report_evolution(ckpts, args.particles, args.max_rows)


if __name__ == '__main__':
    main()
