# -*- coding: utf-8 -*-
"""
Atomic dynamics simulation package developed during my PhD to run simulations
and analyze experimental data. It provides functions to simulate classical
dynamics of atoms in various trapping potentials, involving either the
attractive dipole force or the repulsive ponderomotive force. It also allows
for the computation of related observables such as light shifts and coherence
loss.

The concepts involved are quite specialized. Some relevant explanations can be
found in my thesis, more specifically in appendix D.

More precisely, the package implements:
* Potential computations utilities.
    - Evaluation of non-analytical trapping potentials on a mesh grid
    - Convolution with a charge density (ponderomotive force)
* Sampling of atomic positions and momenta at thermal equilibrium in arbitrary
  potentials.
* Integration of the equations of motion for many atoms at once.
* Simulation of experimental sequences and determination of macroscopic
  observables.
  - Arbitrary sequences
  - Pre-determined sequences
* Many examples to illustrate what is possible with the package.

The code was written while I was rushing to finish writting my thesis.
Although it works and is provided with many examples, the structure is
inelegant and the code is severely lacking documentation. In other words, this
code needs a complete refactoring.
"""

from .simutils import (
    # potential
    Data_Grid,
    build_potential,
    build_force,
    # sampling
    harmonic_sampling,
    rejection_sampling,
    # evolution
    evolution,
    run_simu,
    sim_traj,
    recap_energy,
    recap_interior,
    # physics
    Circ_Prob_Density,
    )

from simpack.potentials import (
    # Generic potential array
    Potential_Array,
    # Gaussian potential
    gbeam_pot_np,
    gbeam_pot_sym,
    build_gbeam_pot, 
    build_gbeam_force,
    gbeam_sampling,
    normal_gbeam_sampler,
    # BoB potential
    BoB_Array,
    )

from .simu_rrtemp import trajectory as traj_rr
from .simu_rrtemp import run_simulation as run_simu_rr

from .simu_gaussdyn import trajectory as traj_gaussdyn
from .simu_gaussdyn import run_simulation as run_simu_gaussdyn

from .simu_bobdyn import trajectory as traj_bobdyn
from .simu_bobdyn import run_simulation as run_simu_bobdyn

from .config.config import *
from .config.physics import ponderomotive_coef

