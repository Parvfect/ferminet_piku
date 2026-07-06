import os
import sys
from absl import logging, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('server_addr', '',
                    help=('Enables multihost calculations if given. '
                          'Server ip address of host node'))
node_id = os.environ['SLURM_NODEID']
visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]



def get_config():
    # Import JAX-dependent stuff ONLY inside function or after init
    from ferminet import base_config
    from ferminet.utils import system
    from ferminet.pbc import envelopes
    import numpy as np
    from pyscf import gto

    cfg = base_config.default()

    MUON_MASS = 206.7682827

    # Set up molecule
    a = 6.74  # Lattice constant in bohr
    cfg.system.particles = (33, 32, 1)
    cfg.system.charges = (-1., -1., 1.)
    cfg.system.masses = (1., 1., MUON_MASS)

    # 2x2x2 diamond supercell (16 atoms), T-relaxed for an unpaired-electron muon
    # at the tetrahedral interstitial (~0.7503, 0.7503, 0.7503)*a. Coordinates are
    # the DFT-relaxed positions in units of `a` (see musr/analysis/diamond_coordinates.py,
    # `atomic_positions_t_relaxed_cubic`). Relaxation is small: max C displacement
    # 0.0414 bohr, a +0.4% T-cage expansion. Geometry IDENTICAL to t_relaxed/pp.py;
    # the ONLY difference in this run is that the muon walkers are seeded at the
    # relaxed T-site (EXP-003b, diamond analogue of silicon t_seeded.py).
    cfg.system.molecule = [
        system.Atom('C', (-0.00060074*a, -0.00060073*a, -0.00060074*a)),
        system.Atom('C', (0.50079041*a, 0.50079041*a, -0.00060075*a)),
        system.Atom('C', (-0.00060075*a, 0.50079041*a, 0.50079041*a)),
        system.Atom('C', (0.50079041*a, -0.00060074*a, 0.50079041*a)),
        system.Atom('C', (0.49925985*a, 1.00130914*a, 0.49925985*a)),
        system.Atom('C', (1.00130915*a, 0.49925984*a, 0.49925984*a)),
        system.Atom('C', (0.49925985*a, 0.49925984*a, 1.00130916*a)),
        system.Atom('C', (1.00130912*a, 1.00130911*a, 1.00130912*a)),
        system.Atom('C', (0.25009483*a, 0.25009485*a, 0.25009484*a)),
        system.Atom('C', (0.75028450*a, 0.75028450*a, 0.24443754*a)),
        system.Atom('C', (0.24443755*a, 0.75028450*a, 0.75028450*a)),
        system.Atom('C', (0.75028450*a, 0.24443755*a, 0.75028449*a)),
        system.Atom('C', (0.75028449*a, 1.25613144*a, 0.75028448*a)),
        system.Atom('C', (1.25613144*a, 0.75028449*a, 0.75028447*a)),
        system.Atom('C', (0.75028448*a, 0.75028449*a, 1.25613145*a)),
        system.Atom('C', (1.25047415*a, 1.25047414*a, 1.25047416*a)),
    ]

    cfg.system.atoms = cfg.system.molecule

    # Pseudopotential setup
    cfg.system.use_pp = True
    cfg.system.pp.symbols = ['C']

    mol = gto.Mole()
    mol.atom = [[atom.symbol, atom.coords] for atom in cfg.system.molecule]

    atoms = list(set([atom.symbol for atom in cfg.system.molecule]))
    pseudo_atoms = cfg.system.pp.symbols if cfg.system.use_pp else []

    mol.basis = {
        atom: cfg.system.pp.basis if atom in pseudo_atoms else 'cc-pvdz'
        for atom in atoms
    }

    mol.ecp = {
        atom: cfg.system.pp.type
        for atom in atoms if atom in pseudo_atoms
    }

    mol.charge = 0
    mol.spin = 0
    mol.unit = 'bohr'
    mol.build()

    cfg.system.pyscf_mol = mol

    # No pretraining for PBC
    cfg.pretrain.method = None

    # Primitive cell of fcc
    cfg.system.pbc.lattice_vectors = np.array([
        [a, a, 0],
        [0, a, a],
        [a, 0, a]
    ])

    cfg.system.pbc.apply_pbc = True
    cfg.network.full_det = False
    cfg.network.ferminet.separate_spin_channels = False
    cfg.system.pbc.min_kpoints = 1

    # Training hyperparameters
    cfg.batch_size = 4096
    cfg.pretrain.iterations = 0
    cfg.log.restore_path = "train"

    # EXP-003b (diamond): seed the muon (last particle) walkers at the RELAXED
    # T-site, i.e. the centroid of the 4 expanded C (#4-#7) = (0.7502845)*a per
    # component (verified: mean of atoms 4-7 = 0.750284*a, 2.9305 bohr equidistant,
    # the +0.4% expanded T-cage). Diamond analogue of silicon t_seeded. Unlike
    # silicon (whose unseeded quantum muon FLED the contracted relaxed cage), the
    # diamond unseeded t_relaxed muon (#12) already HELD the expanded relaxed cage
    # (muonium) -- this run seeds inside for a clean, symmetric comparison with #17
    # and to confirm the held state is not a warm-start/init artifact. bohr
    # cartesian, molecule frame. Same mechanism as bc_seeded.
    cfg.mcmc.muon_init_coord = (0.7502845*a, 0.7502845*a, 0.7502845*a)
    # Seed width below the diamond T-cage scale (2.93 bohr C-T distance, ~2/3 of
    # silicon's), so the compact cloud sits well inside the relaxed cage and relaxes
    # outward to fill the basin. Scaled from silicon's 0.5 by the T-cage size ratio
    # (diamond 2.93 / silicon 4.39 ~ 0.67). init width only sets the START cloud;
    # MCMC+psi govern later motion, so escape (if the relaxed site is not a true
    # min) is still allowed.
    cfg.mcmc.muon_init_width = 0.35  # Gaussian about the relaxed T-site

    return cfg


if __name__ == '__main__':
    flags.FLAGS.mark_as_parsed()

    import jax
    jax.distributed.initialize(
        coordinator_address=FLAGS.server_addr,
        local_device_ids=visible_devices)

    # Now it's safe to import anything using JAX
    import jax.numpy as jnp
    from ferminet import train

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    cfg = get_config()

    # Training config (EXP-003b: T-seeded, fresh network).
    cfg.optim.iterations = 900001
    cfg.log.save_freq = 2000
    cfg.log.save_tfreq = 235
    # Fresh net + first-launch seeding: restore_path == save_path, empty on first
    # launch, so find_last_checkpoint() returns None and the net trains from
    # scratch ("No checkpoint found. Training new model."). Avoids warm-start
    # contamination from the t_relaxed checkpoint. Leave restart.load_data at
    # default True so restarts continue (not re-seed) the chain.
    cfg.log.save_path = "/projects/u6em/parv/diamond/unpaired/t_seeded/pp"
    cfg.log.restore_path = cfg.log.save_path
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"

    train.train(cfg)
