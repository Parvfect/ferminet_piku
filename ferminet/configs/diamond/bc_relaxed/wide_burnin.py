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

    cfg.system.molecule = [
    system.Atom('C', (-0.04975306*a, -0.04975303*a, -0.04975303*a)),
    system.Atom('C', ( 0.29984779*a,  0.29984777*a,  0.2998478*a )),
    system.Atom('C', ( 0.50291589*a,  0.50291587*a,  0.00739201*a)),
    system.Atom('C', ( 0.74755822*a,  0.74755822*a,  0.24270302*a)),
    system.Atom('C', ( 0.50291586*a,  0.007392*a,    0.50291587*a)),
    system.Atom('C', ( 0.74755825*a,  0.24270304*a,  0.74755821*a)),
    system.Atom('C', ( 1.00221291*a,  0.50048928*a,  0.50048927*a)),
    system.Atom('C', ( 1.2486404*a,   0.74998489*a,  0.74998491*a)),
    system.Atom('C', ( 0.00739207*a,  0.50291589*a,  0.50291591*a)),
    system.Atom('C', ( 0.24270298*a,  0.74755822*a,  0.74755819*a)),
    system.Atom('C', ( 0.50048924*a,  1.00221292*a,  0.50048927*a)),
    system.Atom('C', ( 0.74998493*a,  1.24864038*a,  0.74998492*a)),
    system.Atom('C',  ( 0.50048925*a,  0.5004893*a,   1.00221293*a)),
    system.Atom('C', ( 0.74998492*a,  0.74998489*a,  1.24864037*a)),
    system.Atom('C', ( 0.99993799*a,  0.99993794*a,  0.99993791*a)),
    system.Atom('C', ( 1.25091541*a,  1.25091548*a,  1.2509155*a)),
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

    # --- EXP-004: wide muon proposal + long burn-in, unseeded ------------------
    # Identical to bc_relaxed/pp except the MCMC proposal/burn-in schedule. The
    # muon init stays the DEFAULT symmetric carbon-centred Gaussian (no BC seed:
    # muon_init_coord is left None). The muon (last particle) starts with a wide
    # proposal (0.3 ~= its healthy adapted width) while the electrons keep the
    # usual move_width=0.02; each species adapts its own width in the training
    # loop. The long burn-in equilibrates walkers to |psi_fresh|^2 (with the
    # muon mixing broadly) before the optimizer starts localising psi. See
    # experiments/EXP-004_muon_width_burnin_diffusion.md.
    cfg.mcmc.muon_move_width = 0.3   # electrons stay at the default 0.02
    cfg.mcmc.burn_in = 2000

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
    cfg.log.save_freq = 100
    cfg.log.save_tfreq = 235
    # Fresh net: brand-new empty save_path, restore_path == save_path so no
    # off-T bc_relaxed/pp checkpoint can contaminate the run (EXP-004 risk #1).
    # v3: v1 (pp_wide_burnin) and v2 (pp_wide_burnin_v2) both launched from a
    # train.py that predated the muon_move_width block (stale bytecode / launch
    # before the edit landed), so their ckpt_000000 muon width is 0.02, not 0.3
    # -- they never tested H1. Confirmed via tools/muon_width_check.py. v3 runs
    # the verified current code (muon inits at 0.3, no burn-in adaptation) into a
    # fresh dir. Verify the "Initial MCMC width per species: [0.02 0.02 0.3]" log
    # line and ckpt_000000 muon width before trusting the run.
    cfg.log.save_path = "/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_wide_burnin_v3"
    cfg.log.restore_path = cfg.log.save_path
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"

    train.train(cfg)
