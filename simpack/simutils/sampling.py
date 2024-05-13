# -*- coding: utf-8 -*-
"""
Functions for the sampling of initial atomic positions and velocities in a
given potential and at thermal equilibrium.

The sampling functions are:
    - <rejection_sampling>, utility function implementing the rejection
      sampling method for positions in a given potential
    - <harmonic_sampling>, sample positions AND velocities for atoms in
      an harmonic trap at given temperature.


TODO
- gendoc
"""

import numpy as np

from ..config.physics import k_B, m_Rb87

rng = np.random.default_rng()


# =============================================================================
# Functions to sample atomic configurations
# =============================================================================

def harmonic_sampling(nbSamples: int,
                      T: float,
                      omega_x: float,
                      omega_y: float,
                      omega_z: float):
    """
    Draw samples of position and velocity for atoms in an harmonic trap
    with frequencies omega_x, omega_y, omega_z.
    Velocities are drawn from a normal distribution of variance
    k_B * T / m_Rb87.
    
    Note that the vertical is set along y.

    Parameters
    ----------
    nbSamples : int
        Number of samples drawn for the simulation.
    T : float
        The Temperature in Kelvins.
    omega_x, omega_y, omega_z : float
        Frequencies of the harmonic trap.

    Returns
    -------
    samples : np.ndarray
        Atom samples as an array of shape (6, nbSamples). The format is :
            - samples[0, i], samples[1, i], samples[2, i] :
                x, y, z coordinates of atom i respectively
            - samples[3, i], samples[4, i], samples[5, i] :
                v_x, v_y, v_z components of atom i respectively

    """
    ## vertical is along y !
    # standard deviations
    sigma_v = np.sqrt(k_B * T / m_Rb87) # std dev of velocity along an axis
    sigma_x = sigma_v / omega_x # std dev of position along x
    sigma_y = sigma_v / omega_y # std dev of position along y
    sigma_z = sigma_v / omega_z # std dev of position along z
    
    # std deviations of (x, y, z, vx, vy, vz) respectively
    sigma = np.array([sigma_x, sigma_y, sigma_z, sigma_v, sigma_v, sigma_v])
    sigma = sigma.reshape((6, 1))
    
    # row major order !
    samples = rng.normal(0, sigma, (6, nbSamples))
    
    return samples


def rejection_sampling(nbSamples: int,
                       T: float,
                       potential: callable,
                       interior: np.ndarray,
                       loc: tuple,
                       pxsize: tuple)-> np.ndarray:
    """
    Sample positions according to the Maxwell-Boltzmann distribution
    exp(-V/kT) using the rejection sampling method. For a detailed
    explanation of the method, see the wikipedia article.
    
    The algorithm works as follows:
    - Positions are sampled uniformly from the interior by drawing
      a vertex (x0, y0, z0) and a displacement (dx, dy, dz) in [0, 1)^3:
      (x, y, z) = (x0 + dx, y0 + dy, z0 + dz)
    - probabilities p are sampled uniformly in [0, 1)
    - A position (x, y, z) is retained if exp(-V(x, y, z)/kT) >= p
    
    Note: 
        For proper rejection sampling, the potential must be positive.
        For efficient sampling, it is advised to have V = 0 at the minimum.

    Parameters
    ----------
    nbSamples : int
        Number of samples drawn.
    potential : callable x, y, z: V(x, y, z)
        With
        - x, y, z np.ndarray with same shape
        - V(x, y, z) np.ndarray with the same shape as x, y, z
        The potential V must be positive.
    interior : np.ndarray of bool
        Defines the "interior" of the trap, that is, the trapping
        region from which samples are drawn.
        More specifically, if interior[i, j, k] is True, then the
        positions x, y, z = i + r, j + s, k + t for r, s, t in [0, 1)
        are assumed to be in the trapping region.
    loc : tuple (i0, j0, k0)
        The position of the physical origin (position (0, 0, 0)) in the
        array defining the interior, expressed in array coordinates.
        The conversion from array indices to physical positions is:
        position = (index - loc) * scale
        It is not necessarily an integer.
    pxsize : tuple (dx, dy, dz)
        The voxel dimensions of the array defining the interior, in m.
        The conversion from physical positions to array indices is:
        position = (index - loc) * scale
    T : float
        The temperature. It rescales the potential energy.

    Returns
    -------
    spl : 2D np.ndarray
        Atom samples as an array of shape (3, nbSamples). The format is :
            - samples[0, i], samples[1, i], samples[2, i] :
                x, y, z coordinates of atom i respectively

    """
    # Initialization
    spl = np.empty((3, nbSamples), dtype=float) # row major order !
    n = max(10000, 10*nbSamples)
    rspl = np.empty((3, n), dtype=float)
    
    ## Monte-Carlo sampling of the position
    # sample positions
    indices = np.nonzero(interior)
    spln = rng.integers(0, np.size(indices[0]), size=n)
    
    for i in range(3):
        rspl[i, :] = \
            (indices[i][spln] - loc[i] + rng.uniform(size=n)) * pxsize[i]
    # sample rejection threshold
    p = rng.uniform(size=n)
    # rejection sampling
    j = 0 # selected samples counter
    while j < nbSamples:
        sample_filter = p <= np.exp(- potential(*rspl) / (k_B*T))
        if not np.any(sample_filter):
            raise ValueError(
                "rejection sampling: sampling procedure failing, all sampled "
                "positions are being rejected")
        fspl = rspl[:, np.nonzero(sample_filter)[0]] # filtered raw samples
        k = fspl.shape[1]
        if j + k >= nbSamples: # enough filtered samples
            spl[:, j:] = np.copy(fspl[:, 0:nbSamples-j])
            j = nbSamples
        else: # not enough filtered samples
            spl[:, j:j+k] = np.copy(fspl)
            j = j + k
            # draw new samples
            spln = rng.integers(0, np.size(indices[0]), size=n)
            for i in range(3):
                rspl[i, :] = \
                    (indices[i][spln]-loc[i] + rng.uniform(size=n)) * pxsize[i]
            p = rng.uniform(size=n)
    
    return spl

