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

    # 16 Si atoms → 64 valence electrons (with pseudopotential)
    cfg.system.particles = (32, 32)
    cfg.system.charges = (-1., -1.)
    cfg.system.masses = (1., 1.)

    # 2x2x2 diamond supercell (16 atoms)
    cfg.system.molecule = [
        system.Atom('Si', (0, 0, 0)),
        system.Atom('Si', (0.5*a, 0.5*a, 0)),
        system.Atom('Si', (0, 0.5*a, 0.5*a)),
        system.Atom('Si', (0.5*a, 0, 0.5*a)),
        system.Atom('Si', (0.5*a, a, 0.5*a)),
        system.Atom('Si', (a, 0.5*a, 0.5*a)),
        system.Atom('Si', (0.5*a, 0.5*a, a)),
        system.Atom('Si', (a, a, a)),
        system.Atom('Si', (0.25*a, 0.25*a, 0.25*a)),
        system.Atom('Si', (0.75*a, 0.75*a, 0.25*a)),
        system.Atom('Si', (0.25*a, 0.75*a, 0.75*a)),
        system.Atom('Si', (0.75*a, 0.25*a, 0.75*a)),
        system.Atom('Si', (0.75*a, 1.25*a, 0.75*a)),
        system.Atom('Si', (1.25*a, 0.75*a, 0.75*a)),
        system.Atom('Si', (0.75*a, 0.75*a, 1.25*a)),
        system.Atom('Si', (1.25*a, 1.25*a, 1.25*a)),
        system.Atom('H', (0.25*a, 0.25*a, 0.75*a))
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

    cfg.observables.srpd.origin_coord = np.array([0.25*a, 0.25*a, 0.75*a])

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
    cfg.optim.iterations = 100000
    cfg.log.save_freq = 2000
    cfg.log.save_tfreq = 235
    cfg.log.save_path = "/projects/u6em/parv/silicon/T_classical"

    cfg.observables.srpd.calculate = True
    cfg.observables.srpd.use_fixed_origin = True # Relative to coord origin

    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"
    cfg.optim.optimizer = "none"
    cfg.mcmc.fake_energy = True

    train.train(cfg)
