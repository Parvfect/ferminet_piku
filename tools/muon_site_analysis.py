#!/usr/bin/env python3
"""Locate where a quantum muon localised in a FermiNet PBC inference run.

Reads the `positions_*.npy` files dumped by an inference run
(cfg.observables.positions = True) and reports the muon's localisation site,
identifies it as a bond-centre (BC) vs tetrahedral (T) interstitial, and
measures how tightly it is localised.

Why circular statistics? The muon density wraps around the periodic cell, so a
plain Cartesian mean of the samples is meaningless (it averages across periodic
images). Instead we fold positions into the fcc primitive cell, take the
circular mean of the fractional coordinates, and also report the 3D-histogram
peak (mode) as an independent cross-check.

Usage:
    python tools/muon_site_analysis.py <case>
where <case> is 'unrelaxed', 'bc_relaxed_pp', or 'bc_relaxed_pp_new'.
Edit the CASES dict to add more runs. Geometry must match the inference config.
"""
import sys
import glob
import os
import numpy as np

# Lattice constant + fcc-primitive lattice vectors are set per-case via
# set_lattice() before each analysis (diamond a=6.74, silicon a=10.26 bohr).
A = 6.74
L = np.array([[A, A, 0], [0, A, A], [A, 0, A]], dtype=float)
LINV = np.linalg.inv(L)


def set_lattice(a):
    """Set the module-level lattice (constant a, fcc-primitive vectors)."""
    global A, L, LINV
    A = a
    L = np.array([[a, a, 0], [0, a, a], [a, 0, a]], dtype=float)
    LINV = np.linalg.inv(L)


def diamond_fractional_cell(a):
    """Ideal-diamond 2x2 supercell Si/C positions (Cartesian bohr) for const a."""
    frac = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
        [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 0.5, 1], [1, 1, 1],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
        [0.75, 1.25, 0.75], [1.25, 0.75, 0.75], [0.75, 0.75, 1.25], [1.25, 1.25, 1.25],
    ])
    return frac * a

# Total particle count and 0-based muon index for particles=(33,32,1):
# 33 up + 32 down + 1 muon = 66 particles, muon is the last one.
N_PARTICLES = 66
MUON_IDX = 65

# Ideal-diamond C atoms (unrelaxed), a=6.74. Atom 0 = (0,0,0),
# atom 8 = (0.25,0.25,0.25) -> their midpoint (0.125,0.125,0.125)*a is the BC site.
C_UNRELAXED = diamond_fractional_cell(6.74)

# Ideal silicon, a=10.26 (same fractional structure, larger cell).
SI_UNRELAXED = diamond_fractional_cell(10.26)

# DFT-relaxed C atoms around the bond centre (from inference_pp.py).
C_BC_RELAXED = np.array([
    [-0.04975306, -0.04975303, -0.04975303], [0.29984779, 0.29984777, 0.2998478],
    [0.50291589, 0.50291587, 0.00739201], [0.74755822, 0.74755822, 0.24270302],
    [0.50291586, 0.007392, 0.50291587], [0.74755825, 0.24270304, 0.74755821],
    [1.00221291, 0.50048928, 0.50048927], [1.2486404, 0.74998489, 0.74998491],
    [0.00739207, 0.50291589, 0.50291591], [0.24270298, 0.74755822, 0.74755819],
    [0.50048924, 1.00221292, 0.50048927], [0.74998493, 1.24864038, 0.74998492],
    [0.50048925, 0.5004893, 1.00221293], [0.74998492, 0.74998489, 1.24864037],
    [0.99993799, 0.99993794, 0.99993791], [1.25091541, 1.25091548, 1.2509155],
]) * A

# Second DFT-relaxed (open-shell, unpaired-electron) C atoms around the bond
# centre, from inference_pp_relax_2.py. Muon (H) at (0.125,0.125,0.125)*a (BC).
C_BC_RELAXED_2 = np.array([
    [-0.05139189, -0.05139189, -0.05139188], [0.30148672, 0.30148672, 0.30148671],
    [0.50344702, 0.50344703, 0.00524092], [0.74702714, 0.74702712, 0.24485391],
    [0.50344701, 0.00524094, 0.50344701], [0.74702714, 0.24485389, 0.74702714],
    [1.00212309, 0.49931984, 0.49931985], [1.24873038, 0.75115431, 0.75115431],
    [0.00524093, 0.50344703, 0.50344702], [0.24485390, 0.74702712, 0.74702714],
    [0.49931984, 1.00212309, 0.49931985], [0.75115432, 1.24873039, 0.75115431],
    [0.49931984, 0.49931984, 1.00212309], [0.75115432, 0.75115431, 1.24873038],
    [0.99994427, 0.99994428, 0.99994427], [1.25090921, 1.25090920, 1.25090921],
]) * A

CASES = {
    "unrelaxed": dict(
        positions="/projects/u6em/parv/diamond/unpaired/unrelaxed/pp/inference/positions",
        carbons=C_UNRELAXED,
        lattice_a=6.74,
        # BC site = midpoint of the C0-C8 bond (the (0,0,0)-(0.25,0.25,0.25) bond).
        bc_pair=(0, 8),
    ),
    "bc_relaxed_pp": dict(
        positions="/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp/inference/positions",
        carbons=C_BC_RELAXED,
        lattice_a=6.74,
        # The relaxed bond is C0-C1 (atoms pushed apart around the bond centre).
        bc_pair=(0, 1),
    ),
    # No-PP (all-electron) version of bc_relaxed_pp: same BC-relaxed geometry,
    # but 48+1 core electrons per spin restored -> particles=(49,48,1) = 98.
    "bc_relaxed_nopp": dict(
        positions="/projects/u6em/parv/diamond/unpaired/bc_relaxed/nopp/inference/positions",
        carbons=C_BC_RELAXED,
        lattice_a=6.74,
        bc_pair=(0, 1),
        n_particles=98,                 # 49 up + 48 down + 1 muon
        n_up=49,
        n_dn=48,
    ),
    # Second BC-relaxed PP run (open-shell DFT geometry), training job's own
    # inference dir. particles=(33,32,1)=66, muon = last particle.
    "bc_relaxed_pp_relax_2": dict(
        positions="/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_relax_2/inference/positions",
        carbons=C_BC_RELAXED_2,
        lattice_a=6.74,
        # Relaxed bond is C0-C1 (pushed apart about the bond centre).
        bc_pair=(0, 1),
    ),
    "silicon": dict(
        positions="/projects/u6em/parv/silicon_unpaired/unrelaxed/inference/positions",
        carbons=SI_UNRELAXED,
        lattice_a=10.26,
        # Ideal silicon, BC site = midpoint of Si0-Si8 bond.
        bc_pair=(0, 8),
    ),
    # Classical muon: muon is a fixed H nucleus, NOT a particle. Positions hold
    # 65 electrons only (33,32). SRPD uses a fixed origin = the H/muon site.
    "silicon_classical": dict(
        positions="/projects/u6em/parv/silicon_unpaired/classic/inference/positions",
        carbons=SI_UNRELAXED,
        lattice_a=10.26,
        bc_pair=(0, 8),
        n_particles=65,                       # electrons only, no muon particle
        fixed_origin=np.array([0.75, 0.75, 0.75]) * 10.26,  # H at T-site (bohr)
    ),
    # Diamond classical muon, BC-relaxed geometry: H fixed at the BC site
    # (inference_pp_T.py). 65 electrons (33,32), fixed origin = H position.
    "diamond_classical_bc": dict(
        positions="/projects/u6em/parv/diamond/unpaired/classical/bc/inference/positions",
        carbons=C_BC_RELAXED,
        lattice_a=6.74,
        bc_pair=(0, 1),
        n_particles=65,
        fixed_origin=np.array([0.12504759, 0.12504759, 0.1250476]) * 6.74,  # H at BC site
    ),
}


def min_image(d):
    """Minimum-image displacement(s) under the fcc primitive lattice."""
    f = d @ LINV
    f = f - np.round(f)
    return f @ L


def load_muons(positions_dir, n_particles=N_PARTICLES, muon_idx=None):
    if muon_idx is None:
        muon_idx = n_particles - 1
    files = sorted(glob.glob(os.path.join(positions_dir, "positions_*.npy")))
    if not files:
        raise FileNotFoundError(f"no positions_*.npy in {positions_dir}")
    chunks = []
    for fp in files:
        arr = np.load(fp).reshape(-1, n_particles, 3)  # (samples, n_particles, 3)
        chunks.append(arr[:, muon_idx, :])
    return np.concatenate(chunks, axis=0), len(files)


def site_report(name, p, carbons, bc_site):
    """Print nearest carbons and distance to the intended BC site for point p."""
    d = np.array([np.linalg.norm(min_image(p - c)) for c in carbons])
    order = np.argsort(d)
    print(f"\n[{name}] Cartesian {np.round(p, 3)} bohr  cubic-frac {np.round(p / A, 3)}")
    print(f"  distance to intended BC site: {np.linalg.norm(min_image(p - bc_site)):.3f} bohr")
    nearest = [(int(i), round(float(d[i]), 3)) for i in order[:4]]
    print(f"  4 nearest C atoms (idx, bohr): {nearest}")
    # BC site => 2 close carbons at ~half the bond length; T site => 4 roughly equal.
    if d[order[1]] - d[order[0]] > 0.5 and d[order[0]] < 2.3:
        print("  -> looks like a BOND-CENTRE site (2 close carbons)")
    else:
        print("  -> looks like a TETRAHEDRAL (T) site (4 roughly equidistant carbons)")


def muon_spread(case_name):
    """Positional spread (localisation width) of the muon about its centre.

    Uses min-image displacements from the histogram-peak centre so the periodic
    wrap-around doesn't inflate the numbers. Reports per-axis RMS and the
    overall RMS radius sqrt(<|r-r0|^2>)."""
    cfg = CASES[case_name]
    set_lattice(cfg["lattice_a"])
    muons, nfiles = load_muons(cfg["positions"], cfg.get("n_particles", N_PARTICLES))

    # Centre = 3D-histogram peak in fcc-fractional space (robust to wrap-around).
    frac = (muons @ LINV) % 1.0
    H, edges = np.histogramdd(frac, bins=20, range=[(0, 1)] * 3)
    idx = np.unravel_index(np.argmax(H), H.shape)
    peak_frac = np.array([(edges[k][idx[k]] + edges[k][idx[k] + 1]) / 2 for k in range(3)])
    centre = peak_frac @ L

    # Min-image displacement of every muon sample from the centre.
    disp = min_image(muons - centre)             # (Nsamples, 3)
    r = np.linalg.norm(disp, axis=1)
    per_axis_rms = np.sqrt((disp**2).mean(axis=0))
    rms_radius = np.sqrt((r**2).mean())

    print(f"=== case '{case_name}' : muon spread ===")
    print(f"files: {nfiles}, samples: {len(muons)}")
    print(f"centre (hist peak) Cartesian: {np.round(centre, 3)} bohr")
    print(f"per-axis RMS spread (bohr): {np.round(per_axis_rms, 3)}")
    print(f"overall RMS radius sqrt(<r^2>): {rms_radius:.3f} bohr")
    print(f"mean |r|: {r.mean():.3f} bohr   median |r|: {np.median(r):.3f} bohr")
    for q in (50, 90, 95, 99):
        print(f"  {q}% of samples within {np.percentile(r, q):.3f} bohr")


def muon_srpd(case_name, rmax=None, nbins=None, r_search=0):
    """Spin-resolved electron-muon pair density g(r), replicating
    ferminet/observables.py:cal_spin_resolved_pair_density (quantum-muon branch).

    For each walker: rvec = electron_pos - muon_pos (muon = last particle),
    min-image distance under the fcc lattice, then histogram spin-up and
    spin-down electron distances separately into shells normalised by shell
    volume 4/3 pi (r_{i+1}^3 - r_i^3) and by walker count. We average over all
    samples (the live estimator accumulates a sum over steps; shape is identical).

    Defaults match cfg.observables.srpd: rmax=1.0, nbins=200, r_search=0.
    r_search=0 -> min image via rounding (mod(ds+0.5,1)-0.5), same as the code.
    """
    cfg = CASES[case_name]
    set_lattice(cfg["lattice_a"])
    rmax = 1.0 if rmax is None else rmax       # config default
    nbins = 200 if nbins is None else nbins     # config default
    n_up = cfg.get("n_up", 33)                   # up/down electron counts
    n_dn = cfg.get("n_dn", 32)

    # Quantum muon: muon is the last particle (66 total). Classical muon: no
    # muon particle (65 electrons), distances measured from a fixed origin.
    fixed_origin = cfg.get("fixed_origin")
    np_total = cfg.get("n_particles", N_PARTICLES)
    n_elec = np_total if fixed_origin is not None else np_total - 1

    grids = np.linspace(0, rmax, nbins + 1)
    bin_volume = 4 * np.pi / 3.0 * (grids[1:]**3 - grids[:-1]**3)
    centres = grids[:-1] + (grids[1] - grids[0]) / 2

    up_hist = np.zeros(nbins)
    dn_hist = np.zeros(nbins)
    nsamples = 0
    files = sorted(glob.glob(os.path.join(cfg["positions"], "positions_*.npy")))
    for fp in files:
        arr = np.load(fp).reshape(-1, np_total, 3)         # (W, np_total, 3)
        if fixed_origin is not None:
            rvec = arr - fixed_origin                       # all electrons - origin
        else:
            rvec = arr[:, :-1, :] - arr[:, -1:, :]          # electrons - muon (last)
        if r_search == 0:
            f = rvec @ LINV
            f = f - np.round(f)
            rabs = np.linalg.norm(f @ L, axis=-1)          # (W, n_elec)
        else:
            rabs = _min_image_search(rvec.reshape(-1, 3), r_search).reshape(rvec.shape[:-1])
        up_hist += np.histogram(rabs[:, :n_up].ravel(), bins=grids)[0]
        dn_hist += np.histogram(rabs[:, n_up:].ravel(), bins=grids)[0]
        nsamples += arr.shape[0]

    # Per-walker, per-shell-volume normalisation (as in observables.py), then
    # average over all walkers/samples.
    up = up_hist / bin_volume / nsamples
    dn = dn_hist / bin_volume / nsamples

    origin_note = (f"fixed origin {np.round(fixed_origin, 3)}"
                   if fixed_origin is not None else "muon = last particle")
    print(f"=== case '{case_name}' : SRPD g(r) (observables.py method) ===")
    print(f"files: {len(files)}, samples: {nsamples}; {origin_note}; rmax={rmax}, "
          f"nbins={nbins}, r_search={r_search}")
    print(f"integral check: sum(up*binvol)={(up*bin_volume).sum():.3f} (expect ~"
          f"avg # up-e within {rmax} bohr), down={(dn*bin_volume).sum():.3f}")
    out = f"/tmp/srpd_{case_name}_rmax{rmax}.csv"
    np.savetxt(out, np.column_stack([centres, up, dn]),
               header="r_bohr g_up g_down", comments="")
    print(f"wrote g(r) table -> {out}")
    # Compact text summary: peak location and a few sampled points.
    pk = centres[np.argmax(up + dn)]
    print(f"g(r) peak (up+down) at r = {pk:.3f} bohr")
    step = max(1, nbins // 12)
    print("  r(bohr)   g_up      g_down")
    for k in range(0, nbins, step):
        print(f"  {centres[k]:7.3f}  {up[k]:9.4f}  {dn[k]:9.4f}")


def _min_image_search(rvec, radius):
    """Numpy port of min_image_distance_triclinic for r_search>0 (returns norms)."""
    f = rvec @ LINV
    f = (f + 0.5) % 1.0 - 0.5
    rng = np.arange(-radius, radius + 1)
    off = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    cand = (f[:, None, :] + off[None, :, :]) @ L           # (N, noff, 3)
    return np.linalg.norm(cand, axis=2).min(axis=1)


def analyse(case_name):
    cfg = CASES[case_name]
    set_lattice(cfg["lattice_a"])
    carbons = cfg["carbons"]
    i, j = cfg["bc_pair"]
    bc_site = (carbons[i] + carbons[j]) / 2
    bond = np.linalg.norm(min_image(carbons[i] - carbons[j]))

    muons, nfiles = load_muons(cfg["positions"], cfg.get("n_particles", N_PARTICLES))
    print(f"=== case '{case_name}' ===")
    print(f"files: {nfiles}, muon samples: {len(muons)}")
    print(f"intended BC site (mid C{i}-C{j}): {np.round(bc_site, 3)} bohr "
          f"= cubic-frac {np.round(bc_site / A, 3)}; bond {bond:.3f} bohr, "
          f"half-bond {bond / 2:.3f} bohr")

    # Fold into fcc primitive cell -> fractional in [0,1).
    frac = (muons @ LINV) % 1.0

    # Circular mean + concentration R (1=tight, 0=uniform) per fractional axis.
    ang = 2 * np.pi * frac
    cos_m, sin_m = np.cos(ang).mean(0), np.sin(ang).mean(0)
    cmean = (np.arctan2(sin_m, cos_m) / (2 * np.pi)) % 1.0
    R = np.sqrt(cos_m**2 + sin_m**2)
    mean_cart = cmean @ L
    print(f"\nfractional concentration R (tightness): {np.round(R, 3)}")

    # Independent cross-check: peak of a 3D histogram in fractional space.
    H, edges = np.histogramdd(frac, bins=20, range=[(0, 1)] * 3)
    idx = np.unravel_index(np.argmax(H), H.shape)
    peak_frac = np.array([(edges[k][idx[k]] + edges[k][idx[k] + 1]) / 2 for k in range(3)])
    peak_cart = peak_frac @ L

    site_report("circular mean", mean_cart, carbons, bc_site)
    site_report("histogram peak", peak_cart, carbons, bc_site)

    # Localisation: fraction of (subsampled) muons near the histogram peak.
    sub = muons[::50]
    dpeak = np.array([np.linalg.norm(min_image(m - peak_cart)) for m in sub])
    print(f"\nlocalisation about histogram peak (subsample n={len(sub)}):")
    print(f"  within 1.0 bohr: {(dpeak < 1.0).mean():.3f}")
    print(f"  within 2.0 bohr: {(dpeak < 2.0).mean():.3f}")
    print(f"  median distance: {np.median(dpeak):.3f} bohr")


if __name__ == "__main__":
    # usage: muon_site_analysis.py <case> [mode] [rmax] [nbins]
    #   mode = site (default) | spread | srpd
    case = sys.argv[1] if len(sys.argv) > 1 else "unrelaxed"
    mode = sys.argv[2] if len(sys.argv) > 2 else "site"
    if case not in CASES:
        sys.exit(f"unknown case '{case}'; choices: {list(CASES)}")
    if mode == "site":
        analyse(case)
    elif mode == "spread":
        muon_spread(case)
    elif mode == "srpd":
        rmax = float(sys.argv[3]) if len(sys.argv) > 3 else None
        nbins = int(sys.argv[4]) if len(sys.argv) > 4 else None
        muon_srpd(case, rmax=rmax, nbins=nbins)
    else:
        sys.exit(f"unknown mode '{mode}'; choices: site, spread, srpd")
