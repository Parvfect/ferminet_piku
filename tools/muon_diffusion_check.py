#!/usr/bin/env python3
"""Track the muon WALKER-ENSEMBLE spread across training checkpoints.

Motivation
----------
`muon_width_check.py` shows the muon's *proposal* width is healthy (not
collapsed). This tool answers the complementary question: does the muon
*ensemble* ever occupy a diffuse / multi-basin state, or does it localise to a
single basin early and never re-diffuse? A muon that is broad only at the random
initialisation and then collapses within the first few thousand optimiser steps
(while the adaptive proposal width is still tiny) is trapped by an early-time
width-adaptation LAG, not by physics -- see experiments/EXP-001.

We read the stored walker positions (`data.positions`) from each
`qmcjax_ckpt_<step>.npz`, take the last particle (the muon), fold to fractional
coords via the lattice inverse, and report a PBC-safe (circular) ensemble spread
per checkpoint. Optionally, distances to reference cartesian sites (e.g. BC /
off-T / T) label which basin the cloud sits in.

Note: unpickling `data.positions` goes through JAX, so run with
`JAX_PLATFORMS=cpu` on a login/analysis node without a GPU.

Usage
-----
    JAX_PLATFORMS=cpu python tools/muon_diffusion_check.py <run_save_path> \
        --a 6.74 --particles 33,32,1 \
        --site offT=0.474,0.474,0.474 --site BC=0.125,0.125,0.125

`--a` is the lattice constant (bohr); the fcc-primitive lattice
[[a,a,0],[0,a,a],[a,0,a]] is assumed (override with --lattice if needed).
Site coords are given in units of the lattice constant (cubic fractional).
"""
import os
import sys
import glob
import argparse
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np  # noqa: E402


def step_of(f):
    try:
        return int(os.path.basename(f).split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        return -1


def muon_positions(fpath, npart):
    """(N,3) muon (last-particle) walker positions from one checkpoint."""
    d = np.load(fpath, allow_pickle=True)
    data = d['data'].item()
    pos = data['positions'] if isinstance(data, dict) else data.positions
    pos = np.asarray(pos).reshape(-1, npart, 3)
    return np.asarray(pos[:, -1, :], dtype=float)


def circ_std_frac(frac):
    """Circular std per fractional dim; ~0.29 for uniform, ->0 for localised."""
    ang = 2 * np.pi * (frac % 1.0)
    out = np.zeros(3)
    for k in range(3):
        R = np.hypot(np.cos(ang[:, k]).mean(), np.sin(ang[:, k]).mean())
        out[k] = np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0.0)) / (2 * np.pi)
    return out


def min_image_dist(cart, ref, latt, latt_inv):
    dfrac = (cart - ref) @ latt_inv
    dfrac -= np.round(dfrac)
    return np.linalg.norm(dfrac @ latt, axis=1)


def parse_particles(s):
    return [int(x) for x in s.split(',') if x.strip()]


def parse_site(s):
    name, coords = s.split('=')
    return name, np.array([float(x) for x in coords.split(',')])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run save_path (dir of qmcjax_ckpt_*.npz)')
    ap.add_argument('--a', type=float, required=True,
                    help='lattice constant (bohr)')
    ap.add_argument('--particles', type=parse_particles, default=[33, 32, 1],
                    help='species counts (muon last), default 33,32,1')
    ap.add_argument('--lattice', default=None,
                    help='9 comma-separated floats (row-major) overriding the '
                         'default fcc-primitive [[a,a,0],[0,a,a],[a,0,a]]')
    ap.add_argument('--site', action='append', default=[], type=parse_site,
                    help='NAME=fx,fy,fz reference site in lattice-constant units '
                         '(repeatable); reports %% of walkers within --radius')
    ap.add_argument('--radius', type=float, default=1.5,
                    help='bohr radius for the per-site occupancy columns')
    ap.add_argument('--max-rows', type=int, default=40)
    args = ap.parse_args()

    a = args.a
    if args.lattice:
        latt = np.array([float(x) for x in args.lattice.split(',')]).reshape(3, 3)
    else:
        latt = np.array([[a, a, 0.], [0., a, a], [a, 0., a]])
    latt_inv = np.linalg.inv(latt)
    npart = sum(args.particles)
    sites = [(n, f * a) for n, f in args.site]

    files = sorted(glob.glob(os.path.join(args.run, 'qmcjax_ckpt_*.npz')),
                   key=step_of)
    files = [f for f in files if step_of(f) >= 0]
    if not files:
        print(f'no checkpoints under {args.run}', file=sys.stderr)
        sys.exit(1)
    # dense early, subsample late
    early = [f for f in files if step_of(f) <= 40000]
    rest = [f for f in files if step_of(f) > 40000]
    keep_rest = max(1, len(rest) // max(1, args.max_rows - len(early)))
    sel = sorted(set(early + rest[::keep_rest]), key=step_of)

    print(f'run: {args.run}  ({len(files)} ckpts; sampling {len(sel)})')
    site_hdr = ''.join(f'{f"%<{args.radius}b_{n}":>13}' for n, _ in sites)
    hdr = f'{"step":>8}{"Nwalk":>8}{"frac_std(x,y,z)":>24}{"spread_bohr":>12}' + site_hdr
    print(hdr)
    print('-' * len(hdr))
    for f in sel:
        try:
            mu = muon_positions(f, npart)
        except Exception as e:  # pylint: disable=broad-except
            print(f'{step_of(f):>8}  skip: {e}')
            continue
        cs = circ_std_frac(mu @ latt_inv)
        spread = np.linalg.norm(cs) * a
        cells = ''
        for _, ref in sites:
            pct = 100 * (min_image_dist(mu, ref, latt, latt_inv) < args.radius).mean()
            cells += f'{pct:>13.1f}'
        fs = f'{cs[0]:.3f},{cs[1]:.3f},{cs[2]:.3f}'
        print(f'{step_of(f):>8}{len(mu):>8}{fs:>24}{spread:>12.2f}{cells}')
    print('\nspread_bohr = |circular std| x a. Large only at step 0 then a fast '
          'drop = diffuse-only-at-init (early-time trap); flat-small throughout '
          '= never diffused. See experiments/EXP-001.')


if __name__ == '__main__':
    main()
