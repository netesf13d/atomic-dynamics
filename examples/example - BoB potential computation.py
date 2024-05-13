# -*- coding: utf-8 -*-
"""
This script illustrates the computation of a bottle beam intensity profile and
the convolution with a circular electron.

The effect of this convolution on the trapping potential is described in
chapter 5.2 of my thesis. https://theses.hal.science/tel-04551702
"""

import sys
from pathlib import Path

import numpy as np

# simpack in parent directory
pdir = Path(__file__).resolve().parents[1]
if not (simd := str(pdir)) in sys.path:
    sys.path.append(simd)
import simpack as sim


# =============================================================================
# ########## CONFIGURATION ##########
# =============================================================================

FLOAT_TYPE = np.float64
POT_PATH = "./potentials/"
axes = ("x", "y", "z")

## Bottle beam characteristics
bob_defaults = {
    'wavelength': 8.21e-07, # m
    'focal_length': 0.0163, # m
    'I_waist': 0.004,
    'I_offset': 0.1,
    'pupil_radius': 0.0055, # m
    'r_bob': 0.0035} # m


## Parameters for precise computation : requires 16Gb of memory
# shape of the BoB intensity profile, memory requirement scales as shx*shy*shz
shx, shy, shz = 2**9, 2**9, 2**8
# pixel size of the BoB intensity profile; physical extension is pxsz * sh
pxsz = (0.008 * 1e-6, 0.008 * 1e-6, 0.04 * 1e-6) # m
# pixel size of the circular electron probability distribution
pd_pxsz = (5e-9, 5e-9, 2e-9) # m
# pixel size of the resulting BoB potential
pot_pxsz = (0.02 * 1e-6, 0.02 * 1e-6, 0.06 * 1e-6) # m


## Parameters for quick computation : requires 64Mb of memory
shx, shy, shz = 2**6, 2**6, 2**6 # memory requirement scales as shx*shy*shz
pxsz = (0.06 * 1e-6, 0.06 * 1e-6, 0.140 * 1e-6) # m
pd_pxsz = (10e-9, 10e-9, 5e-9) # m
pot_pxsz = (0.06 * 1e-6, 0.06 * 1e-6, 0.14 * 1e-6)

# =============================================================================
# ########## Trapping potential for simulations ##########
# =============================================================================

print("\n========== Compute trapping potential for simulations ==========\n")

# print("Computing and analyzing BoB intensity profile...")
### Compute BoB intensity profile normalized to unit incident beam power
bob = sim.BoB_Array(shape=(shx, shy, shz),
                    pxsize=pxsz,
                    bobParameters=bob_defaults,
                    dtype=FLOAT_TYPE,
                    xfrac_exp=4) # memory requirement scales as 4**xfrac_exp
## The intensity profile is stored in a Data_Grid object
print(bob.bob)


### Create a Potential_Array to analyze the potential
bob = sim.Potential_Array(source=bob.bob) # instanciation from a Data_Grid
bob.plot_section(posIdx=(32, 32, 32), entry="potential")
## compute relevant properties of the trap
bob.analyze("threshold") # trap depth
bob.analyze("interior") # trapping region
bob.plot_section(entry="interior")
bob.analyze("curvature") # trap curvature at the center
## save...
bob.save(POT_PATH + "BoB_pot.npz")
## ...and load the data
bob = sim.Potential_Array(source=POT_PATH + "BoB_pot.npz")


## Compute the potential by convolution with the state probability density.
ns = [50, 52] # consider states 50C and 52C
axis = 0 # quantization axis along x (the BoB propagates along z)

for n in ns:
    # print(f"Computing BoB trap n={n}...")
    ## Compute probability density
    pd = sim.Circ_Prob_Density(n=n, norm=1.)
    pd = pd.prob_density_array(pd_pxsz,
                               eval_mode="analytical",
                               optimize_shape=True)
    pd = sim.Data_Grid(**pd)
    # coarse-grained yet non-trivial probability density
    pd_ = sim.Potential_Array(source=pd)
    pd_.plot_section()
    
    ## Convolve probability density with potential
    # the probability density pixel sizes must divide those of the intensity profile
    pot = bob.compute_potential(pxsize=pot_pxsz,
                                probability_density=pd,
                                axis=axis,
                                symmetries=(0, 1, 2)) # returns a Data_Grid
    pot = sim.Potential_Array(pot)
    pot.plot_section()
    pot.analyze("threshold")
    pot.analyze("interior")
    pot.analyze("curvature")
    pot.save(POT_PATH + f"BoB_pot_{n}C_{axes[axis]}.npz")

