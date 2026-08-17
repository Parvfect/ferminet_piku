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

    # --- EXP-005: FROZEN adapter (wide muon held wide all run), unseeded --------
    # Follow-up to EXP-004 (pp_wide_burnin_v3), which showed a one-shot wide muon
    # init does NOT un-trap the muon: the adaptive width collapsed 0.30 -> 0.072
    # within ~1500 steps (adapter drives the muon to ~50% acceptance, which the
    # sharpening psi puts at ~0.08 bohr), the ensemble committed to a single
    # basin by step ~700, and the muon localised at a T-type interstitial 3.2
    # bohr from BC (H1 falsified). See experiments/EXP-005.
    #
    # EXP-005 removes MCMC mobility as a confound entirely by FREEZING the
    # proposal width for the whole run: the muon stays at its wide 0.3 bohr
    # proposal (which equilibrated to a 2.6-bohr cell-spanning cloud during v3's
    # burn-in) for every optimiser step in the selection window, so the muon is
    # maximally mobile throughout. Logic: if the muon STILL localises off-BC with
    # a permanently-wide proposal, the trap is an OPTIMISATION / representability
    # artifact, not a sampling one -> next lever is an envelope-level muon bias,
    # not a slower adapter. If it un-traps to BC, sampling mobility WAS the lever.
    #
    # Freeze mechanism (ZERO code change): adapt_frequency is set larger than the
    # scientifically-relevant window so update_mcmc_width never fires
    # (`if t > 0 and t % adapt_frequency == 0`). We use 100000 (NOT ~1e9) because
    # the pmoves buffer is allocated np.zeros((nspecies, adapt_frequency)) in
    # train.py -- 1e9 would try to allocate ~24 GB and OOM at startup, whereas
    # 100000 is a ~2.4 MB buffer and still freezes the width for the first
    # 100k steps (20x past the ~5k selection window; the muon site is long
    # decided before the adapter's first possible firing at t=100000).
    #
    # A ~25% muon acceptance at width 0.3 is expected and FINE: it is near the
    # RWM efficiency optimum (~0.234), Metropolis is unbiased for any proposal
    # width, and 0.3 ~= 1-2x the localised muon's per-axis density width, so it
    # remains an efficient in-site sampler even if psi localises. Electrons are
    # frozen at move_width=0.07 (near their v3-adapted 0.084 equilibrium) so all
    # species sit in a healthy sampling regime for the whole frozen run.
    cfg.mcmc.muon_move_width = 0.3     # muon proposal, held wide (frozen)
    cfg.mcmc.move_width = 0.07         # electron proposal, frozen near equilibrium
    cfg.mcmc.adapt_frequency = 100000  # >> selection window => width never adapts
    cfg.mcmc.burn_in = 2000            # equilibrate walkers to |psi_fresh|^2

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
    # off-T bc_relaxed/pp checkpoint (or any v3 checkpoint) can contaminate the
    # run (EXP-004 confound #1). Pre-launch check on the compute node before
    # trusting the run (same protocol as v3):
    #   * log says "No checkpoint found. Training new model."
    #   * log prints "Initial MCMC width per species: [0.07 0.07 0.3]"
    #   * ckpt_000000 has muon width 0.3, electrons 0.07, and the muon walkers
    #     are a broad diffuse cloud (NOT pinned to a site).
    #   * confirm the width STAYS 0.3/0.07 at ckpt_000100..002000 (never adapts).
    cfg.log.save_path = "/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_wide_burnin_frozen"
    cfg.log.restore_path = cfg.log.save_path
    cfg.optim.reset_if_nan = True
    cfg.optim.laplacian = "folx"

    train.train(cfg)
