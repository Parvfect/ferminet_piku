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
    cfg.system.particles = (33, 32, 1)
    cfg.system.charges = (-1., -1., 1.)
    cfg.system.masses = (1., 1., MUON_MASS)

    # 2x2x2 diamond supercell (16 atoms), T-relaxed for an unpaired-electron muon
    # at the tetrahedral interstitial (0.75, 0.75, 0.75)*a. Coordinates are the
    # DFT-relaxed positions in units of `a` (see musr/analysis/silicon_coordiantes.py,
    # `atomic_positions_t_relaxed`). Relaxation is small: only the 4 Si coordinating
    # the muon move (~0.0487 bohr inward), a -1.1% T-cage contraction.
    # Geometry MUST match training config t_relaxed.py exactly.
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

    mol.charge = 0
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

    # Inference config (EXP-003): restore the trained silicon t_relaxed quantum
    # checkpoint (#10, job family muon_silicon_q_t_relaxed) and sample muon
    # positions. No optimisation (optimizer="none"); positions for the first 1000
    # steps are written to <save_path>/positions/positions_{t}.npy. The SRPD
    # bound-state (muonium) check is computed post-hoc from these positions by
    # tools/muon_site_analysis.py (case 'silicon_t_relaxed', mode 'srpd').
    cfg.optim.iterations = 10000000
    cfg.log.save_freq = 2000000
    cfg.log.save_tfreq = 235000
    cfg.log.restore_from_checkpoint = True
    # Trained checkpoints live in the #10 training save dir (latest at launch).
    cfg.log.restore_path = "/projects/u6em/parv/silicon_unpaired/t_relaxed"
    # Fresh inference output dir: find_last_checkpoint(save_path) returns None and
    # falls back to restore_path. Positions land in <save_path>/positions.
    cfg.log.save_path = "/projects/u6em/parv/silicon_unpaired/t_relaxed/inference"
    cfg.observables.positions = True
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"
    cfg.optim.optimizer = "none"
    cfg.mcmc.fake_energy = True

    train.train(cfg)
