#!/usr/bin/env python3
"""Extended-radius spin-resolved pair density: bound state vs lost-to-bands.

The default SRPD in tools/muon_site_analysis.py only goes to rmax=1 bohr, which
sees the contact (1s) region. That cannot distinguish a *loose* bound state from
"electron lost to the bands": both can look spin-balanced inside 1 bohr. This
script extends g(r) out toward the Wigner-Seitz inscribed radius and separates
the two effects a short-range window conflates:

  1. A spin-SYMMETRIC screening cloud (any +1 muon has one) -> g_up ~ g_down,
     enhanced near the muon, integrating to ~1 screening electron. Diamagnetic.
  2. The NET SPIN density (the muonium discriminator): cumulative N_up(r)-N_dn(r).
     - Lost to bands: tracks the uniform-band line (the lone doublet up-electron
       spread evenly over the cell, ~ rho_diff * 4/3 pi r^3).
     - (Loose) bound state: a LOCALIZED net-spin excess that rises ABOVE the
       uniform line and plateaus toward ~+1 at a radius BELOW the lattice scale.

Uses the same per-case geometry/positions as muon_site_analysis.CASES, so it
works for any quantum-muon case (silicon_bc_relaxed, silicon_t_relaxed, ...).

Usage (ferminet-piku env):
    python tools/srpd_extended_radius.py <case> [rmax] [nbins] [stride]
e.g. python tools/srpd_extended_radius.py silicon_bc_relaxed 6.0 120 4
"""
import sys
import numpy as np

from muon_site_analysis import CASES, iter_frames


def main(case_name, rmax=6.0, nbins=120, stride=4):
    if case_name not in CASES:
        sys.exit(f"unknown case '{case_name}'; choices: {list(CASES)}")
    cfg = CASES[case_name]
    fixed_origin = cfg.get("fixed_origin")
    # Quantum runs: muon is the last walker particle (distances measured to it).
    # Classical runs: muon is a fixed point charge (no walker particle); distances
    # are measured from that fixed origin and ALL particles are electrons.

    a = cfg["lattice_a"]
    L = np.array([[a, a, 0], [0, a, a], [a, 0, a]], float)
    LINV = np.linalg.inv(L)
    Vcell = abs(np.linalg.det(L))
    n_up = cfg.get("n_up", 33)
    n_dn = cfg.get("n_dn", 32)
    rho_up, rho_dn = n_up / Vcell, n_dn / Vcell

    # WS inscribed radius = half the shortest lattice/translation vector; the
    # max r for which full spherical shells stay inside the min-image cell.
    vecs = [L[i] for i in range(3)] + [L[i] - L[j]
                                       for i in range(3) for j in range(3) if i != j]
    ws = min(np.linalg.norm(v) for v in vecs) / 2
    if rmax > ws:
        print(f"WARNING: rmax {rmax} exceeds WS inscribed radius {ws:.2f}; "
              "shells beyond it under-count and dip below bulk.")

    grids = np.linspace(0, rmax, nbins + 1)
    shellV = 4 * np.pi / 3 * (grids[1:]**3 - grids[:-1]**3)
    vol = 4 * np.pi / 3 * grids[1:]**3
    cen = grids[:-1] + (grids[1] - grids[0]) / 2
    up = np.zeros(nbins)
    dn = np.zeros(nbins)
    ns = 0
    nframes = 0
    # Frames come from inference dumps or, for checkpoint-only cases, the
    # training checkpoints (see muon_site_analysis.iter_frames). stride
    # subsamples frames (keep every stride-th).
    for k, arr in enumerate(iter_frames(cfg)):
        if k % stride:
            continue
        nframes += 1
        if fixed_origin is not None:
            rvec = arr - fixed_origin              # all electrons - fixed muon site
        else:
            rvec = arr[:, :-1, :] - arr[:, -1:, :]  # electrons - muon (last particle)
        f = rvec @ LINV
        f -= np.round(f)
        r = np.linalg.norm(f @ L, axis=-1)         # (W, n_elec)
        up += np.histogram(r[:, :n_up].ravel(), bins=grids)[0]
        dn += np.histogram(r[:, n_up:].ravel(), bins=grids)[0]
        ns += arr.shape[0]

    g_up = up / shellV / ns
    g_dn = dn / shellV / ns
    Nup = np.cumsum(up) / ns
    Ndn = np.cumsum(dn) / ns
    netspin = Nup - Ndn
    netspin_unif = (rho_up - rho_dn) * vol         # delocalized 1 e^- spread uniformly

    print(f"=== case '{case_name}' : extended-radius SRPD ===")
    print(f"Vcell={Vcell:.1f} bohr^3  rho_up={rho_up:.5f} rho_dn={rho_dn:.5f} e/bohr^3")
    print(f"WS inscribed radius (max valid r) ~ {ws:.2f} bohr")
    print(f"frames used: {nframes} (stride {stride})  samples: {ns}\n")
    print(" r     g_up/bulk g_dn/bulk | N_up   N_dn   netSpin  netSpin_unif  localized_excess")
    for k in range(0, nbins, max(1, nbins // 20)):
        print(f"{cen[k]:4.2f}    {g_up[k]/rho_up:5.2f}     {g_dn[k]/rho_dn:5.2f}   |"
              f" {Nup[k]:5.2f} {Ndn[k]:5.2f}  {netspin[k]:+.3f}    {netspin_unif[k]:+.3f}"
              f"      {netspin[k] - netspin_unif[k]:+.3f}")
    print("\nVERDICT GUIDE: localized_excess ~0 (tracks uniform) => electron lost "
          "to bands (diamagnetic). A localized_excess rising to ~+1 and plateauing "
          "below the lattice scale => a (loose) bound state / muonium.")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "silicon_bc_relaxed"
    rmax = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    nbins = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    main(case, rmax, nbins, stride)
