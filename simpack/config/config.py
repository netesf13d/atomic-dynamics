# -*- coding: utf-8 -*-
"""
Configuration for the simpack package.
"""

import numpy as np


# =============================================================================
# SIMULATION PARMETERS
# =============================================================================

# Release-recapture elementary timestep
RR_TIMESTEP = 0.1 * 1e-6 # s

# Gauss dynamics elementary timestep
GAUSSDYN_TIMESTEP = 0.1 * 1e-6 # s

# Bob dynamics elementary timestep
BOBDYN_TIMESTEP = 0.2 * 1e-6 # s


# =============================================================================
# BoB generation parameters
# =============================================================================

# float type used for BoB profile computation
# The complex type is accordingly complex64 for float32 and
# complex128 for float64
FLOAT_TYPE = np.float64

# log2 of the fraction of the FFT computation in which the BoB profile ends up
# For a BoB computation on an array of shape (2**nx, 2**ny), the BoB profile
# is essentially contained in a centered sub-array of shape
# (2**(nx-XFRAC), 2**(ny-XFRAC))
XFRAC = 4

# Number of dichotomy iterations for threshold computation
NB_THR_ITERATIONS = 16



