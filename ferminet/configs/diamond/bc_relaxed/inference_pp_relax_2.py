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

    # Carbon positions from the BC muon run relaxed in DFT with an unpaired
    # electron (open-shell). Cubic-coordinate units (fractions of a); multiply by
    # a for cartesian bohr. Muon (H) at 0.125, 0.125, 0.125.
    # Geometry MUST match training config pp_relax_2.py exactly.
    cfg.system.molecule = [
    system.Atom('C', ( -0.05139189*a,  -0.05139189*a,  -0.05139188*a)),
    system.Atom('C', (  0.30148672*a,   0.30148672*a,   0.30148671*a)),
    system.Atom('C', (  0.50344702*a,   0.50344703*a,   0.00524092*a)),
    system.Atom('C', (  0.74702714*a,   0.74702712*a,   0.24485391*a)),
    system.Atom('C', (  0.50344701*a,   0.00524094*a,   0.50344701*a)),
    system.Atom('C', (  0.74702714*a,   0.24485389*a,   0.74702714*a)),
    system.Atom('C', (  1.00212309*a,   0.49931984*a,   0.49931985*a)),
    system.Atom('C', (  1.24873038*a,   0.75115431*a,   0.75115431*a)),
    system.Atom('C', (  0.00524093*a,   0.50344703*a,   0.50344702*a)),
    system.Atom('C', (  0.24485390*a,   0.74702712*a,   0.74702714*a)),
    system.Atom('C', (  0.49931984*a,   1.00212309*a,   0.49931985*a)),
    system.Atom('C', (  0.75115432*a,   1.24873039*a,   0.75115431*a)),
    system.Atom('C', (  0.49931984*a,   0.49931984*a,   1.00212309*a)),
    system.Atom('C', (  0.75115432*a,   0.75115431*a,   1.24873038*a)),
    system.Atom('C', (  0.99994427*a,   0.99994428*a,   0.99994427*a)),
    system.Atom('C', (  1.25090921*a,   1.25090920*a,   1.25090921*a)),
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

    # Inference config: restore the trained bc_relaxed_2 (pp_relax_2) checkpoint and
    # sample muon positions. No optimisation (optimizer="none"); positions for the
    # first 1000 steps are written to <save_path>/positions/positions_{t}.npy.
    cfg.optim.iterations = 10000000
    cfg.log.save_freq = 2000000
    cfg.log.save_tfreq = 235000
    cfg.log.restore_from_checkpoint = True
    # Trained checkpoints live here (training job 5344740 writes them directly).
    cfg.log.restore_path = "/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_relax_2"
    # Fresh inference output dir: find_last_checkpoint(save_path) returns None and
    # falls back to restore_path. Positions land in <save_path>/positions.
    cfg.log.save_path = "/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_relax_2/inference"
    cfg.observables.positions = True
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"
    cfg.optim.optimizer = "none"
    cfg.mcmc.fake_energy = True

    train.train(cfg)
