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

    # Silicon lattice constant (bohr)
    a = 10.26  # ~5.43 Å

    # Classical muon: the muon is a FIXED nucleus (the 'H' atom below at the
    # DFT-relaxed T position), NOT a sampled quantum particle. Only the 65
    # electrons (33 up, 32 down — matching the quantum (33,32,1) run) are sampled.
    cfg.system.particles = (33, 32)
    cfg.system.charges = (-1., -1.)
    cfg.system.masses = (1., 1.)

    # 2x2x2 diamond supercell (16 Si), T-relaxed for an unpaired-electron muon.
    # Coordinates are the DFT-relaxed positions in units of `a`
    # (see musr/analysis/silicon_coordiantes.py, `atomic_positions_t_relaxed`).
    # The muon (final 'H') sits at the DFT-relaxed tetrahedral site (~0.75*a).
    cfg.system.molecule = [
        system.Atom('Si', (-0.00067502*a, -0.00067507*a, -0.00067507*a)),
        system.Atom('Si', (0.50067447*a, 0.50067444*a, -0.00067510*a)),
        system.Atom('Si', (-0.00067507*a, 0.50067440*a, 0.50067444*a)),
        system.Atom('Si', (0.50067443*a, -0.00067511*a, 0.50067443*a)),
        system.Atom('Si', (0.50273842*a, 0.99725963*a, 0.50273848*a)),
        system.Atom('Si', (0.99725956*a, 0.50273849*a, 0.50273853*a)),
        system.Atom('Si', (0.50273848*a, 0.50273854*a, 0.99725958*a)),
        system.Atom('Si', (0.99725948*a, 0.99725952*a, 0.99725955*a)),
        system.Atom('Si', (0.24999972*a, 0.24999967*a, 0.24999969*a)),
        system.Atom('Si', (0.74999906*a, 0.74999903*a, 0.24997919*a)),
        system.Atom('Si', (0.24997923*a, 0.74999905*a, 0.74999903*a)),
        system.Atom('Si', (0.74999907*a, 0.24997917*a, 0.74999902*a)),
        system.Atom('Si', (0.74999898*a, 1.25001884*a, 0.74999901*a)),
        system.Atom('Si', (1.25001885*a, 0.74999897*a, 0.74999902*a)),
        system.Atom('Si', (0.74999903*a, 0.74999902*a, 1.25001882*a)),
        system.Atom('Si', (1.24999827*a, 1.24999842*a, 1.24999838*a)),
        system.Atom('H', (0.74999906*a, 0.74999903*a, 0.74999906*a)),
    ]

    cfg.system.atoms = cfg.system.molecule

    # Pseudopotential setup
    cfg.system.use_pp = True
    cfg.system.pp.symbols = ['Si']

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

    mol.charge = 1
    mol.spin = 0
    mol.unit = 'bohr'
    mol.build()

    cfg.system.pyscf_mol = mol

    # No pretraining for PBC
    cfg.pretrain.method = None

    # Supercell lattice vectors (kept same structure)
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

    # Training config
    cfg.optim.iterations = 900001
    cfg.log.save_freq = 2000
    cfg.log.save_tfreq = 235
    cfg.log.save_path = "/projects/u6em/parv/silicon_unpaired/classical/t_relaxed"
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"

    train.train(cfg)
