from ferminet import base_config
from ferminet.utils import system
from pyscf import gto
import numpy as np

def get_config():
  cfg = base_config.default()

  # Lattice constant for carbon (diamond) in bohr
  a = 6.74  # ~3.567 Å
  MUON_MASS = 206.7682827

  # 2x2x2 supercell: 8 primitive cells × 2 atoms = 16 C atoms
  # 4 valence electrons per C atom (pseudopotential) = 64 valence electrons
  # 32 spin-up, 32 spin-down
  cfg.system.particles = (32, 32, 1)
  cfg.system.charges = (-1., -1., 1.)
  cfg.system.masses = (1., 1., MUON_MASS)

  # Build 2x2x2 supercell atom positions
  # Primitive cell FCC lattice vectors (used to generate positions)
  a1 = np.array([0,     a/2,  a/2])
  a2 = np.array([a/2,   0,    a/2])
  a3 = np.array([a/2,   a/2,  0  ])

  # Basis atoms in the primitive cell (fractional of primitive vectors)
  basis = [
      np.array([0, 0, 0]),
      np.array([0.25*a, 0.25*a, 0.25*a])  # Cartesian offset
  ]

  atoms = []
  for i in range(2):
    for j in range(2):
      for k in range(2):
        translation = i*a1 + j*a2 + k*a3
        for b in basis:
          pos = translation + b
          atoms.append(system.Atom('C', tuple(pos)))

  cfg.system.molecule = atoms

  cfg.pretrain.method = None

  # 2x2x2 supercell lattice vectors (double each FCC primitive vector in each direction)
  # The supercell matrix is 2*I applied to the primitive cell,
  # so supercell vectors = a1+a2+a3 combinations scaled by 2
  # Conventional: supercell = 2 * conventional cubic cell vectors
  cfg.system.pbc.lattice_vectors = np.array([[0,   a,   a  ],
                                             [a,   0,   a  ],
                                             [a,   a,   0  ]])

  cfg.system.use_pp = True
  cfg.system.pp.symbols = ['C']

  mol = gto.Mole()
  mol.atom = [[atom.symbol, atom.coords] for atom in cfg.system.molecule]
  atom_symbols = list(set([atom.symbol for atom in cfg.system.molecule]))
  pseudo_atoms = cfg.system.pp.symbols if cfg.system.use_pp else []
  mol.basis = {
      atom: cfg.system.pp.basis if atom in pseudo_atoms else 'cc-pvdz'
      for atom in atom_symbols
  }
  mol.ecp = {
      atom: cfg.system.pp.type
      for atom in atom_symbols if atom in pseudo_atoms
  }
  mol.charge = 0
  mol.spin = 0
  mol.unit = 'bohr'
  mol.build()

  cfg.system.pyscf_mol = mol
  cfg.system.pbc.apply_pbc = True
  cfg.network.full_det = False
  cfg.batch_size = 4096
  cfg.pretrain.iterations = 0
  cfg.log.restore_path = "train"

  return cfg
