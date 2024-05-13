# -*- coding: utf-8 -*-
"""
Functions to run simulations of release-recapture experiments in gaussian
tweezers, and fit the temperature.

A release-recapture experiment destined to measure atom temperature
consists in switching the tweezers off for variable delays ti:
Tweezers ON |<--OFF-- t --OFF-->| Tweezers ON
The sequence is characterized by a list [t1, t2, t3, ..., tk] of delays,
with t1 < t2 < t3 < .. < tk.

Note that the module implements only the computation of recapture
probability.

The functions are:
    - <trajectory>, to simulate atomic trajectories in a release-recapture
        sequence
    - <run_simulation>, to run a simulation of a release-recapture sequence

"""

from typing import Callable

import numpy as np

from .simutils import (evolution, recap_energy)
from .config.config import RR_TIMESTEP as dt

# =============================================================================
# Run simulation
# =============================================================================

def trajectory(run: tuple[float, float, float],
               force: tuple[Callable, Callable, Callable],
               isamples: np.ndarray):
    """
    Compute the trajectory of a set of atom samples in a given run of
    a trap oscillation experiment.

    The evolving positions and velocities of the samples are given at each
    time step. Note that this generates a lot of data, hence this function
    should be used for a small number of samples.

    Parameters
    ----------
    run : tuple (T0, tau, T'0)
        Parameters that characterize the experimental run:
        - T0 : float, portion of the trajectory computed before the
               actual experiment
        - tau : float, duration of release
        - T'0 : float, portion of the trajectory computed after the
                actual experiment (the recapture)
    force : 3-tuple of callable, 3 arguments (x, y, z)
        Functions (Fx, Fy, Fz) that return the components of the force
        as a function of the position (x, y, z).
    isamples : np.ndarray
        Atom samples as an array of shape (6, nbSamples). The format is :
            - ispl[0, i], ispl[1, i], ispl[2, i] :
                x, y, z coordinates of atom i respectively
            - ispl[3, i], ispl[4, i], ispl[5, i] :
                v_x, v_y, v_z components of atom i respectively

    Returns
    -------
    times : 1D np.ndarray
        The times (in seconds) at which the trajectory points are computed.
    traj : 3D np.ndarray
        Atomic samples positions and velocity at each timestep.
        traj_spl[i, :, :] = trajectory samples at times[i]
        (same format as isamples)

    """
    # Get times of each trajectory step and related time intervals
    times = np.array([0], dtype=float)
    steps = []
    for t in run:
        N = int(np.ceil(round(t / dt, ndigits=5))) # nb of atomic timesteps
        t0 = times[-1]
        times = np.append(times, t0 + np.linspace(dt, N*dt, N, endpoint=True))
        times[-1] = t0 + t
        steps.append(np.full(N, dt, dtype=float))
        try:
            steps[-1][-1] = t - (N-1)*dt
        except IndexError: # edge case t=0
            pass
    # assemble forces
    forces = (force, None, force)
    # Compute trajectory
    traj = np.empty((len(times), 6, isamples.shape[1]), dtype=float)
    traj[0, :, :] = np.copy(isamples)
    i = 0
    for k, T in enumerate(steps):
        F = forces[k]
        for step in T:
            traj[i+1, :, :] = np.copy(traj[i, :, :])
            evolution(traj[i+1, :, :], step, force=F)
            i += 1
    return times, traj


def run_simulation(sequence: tuple,
                   potential: Callable,
                   samples: np.ndarray,
                   recap_thr: float = 0.)-> np.ndarray:
    """
    From a set of samples, generate a simulated curve of recapture
    probability at given delays, for a sequence of release-recapture.

    From the list of delays [t1, t2, ..., tk], where k = nbDelays, the
    samples are left to evolve for nbDelays successive delays
    t1, t2-t1, t3-t2, ..., t(k-1)-tk. The recapture probability is computed
    at each step.

    The recapture is determined by the criterion of mechanical energy
    below threshold in the presence of the trapping potential.

    Parameters
    ----------
    sequence : tuple (delays,)
        delays : np.ndarray, tweezers off durations [t1, t2, t3, .., tk]
                 with t1 < t2 < t3 < .. < tk.
    potential : callable, 3 arguments (x, y, z)
        Function that returns the trapping potential as a function of
        the position (x, y, z).
    samples : np.ndarray
        Atom samples as an array of shape (6, nbSamples). The format is :
            - samples[0, i], samples[1, i], samples[2, i] :
                x, y, z coordinates of atom i respectively
            - samples[3, i], samples[4, i], samples[5, i] :
                v_x, v_y, v_z components of atom i respectively
    recap_thr : float, optional
        Recapture threshold. Atoms are recptured if Em < recap_thr.
        The default is 0.

    Raises
    ------
    ValueError
        If the sequence delays are not increasing, ie some successive
        delay is negative. Negative time evolution is ill defined.

    Returns
    -------
    th_pRecap : np.ndarray of shape (nbDelays,)
        Simulated recapture probabilities from samples at delays
        t1, t2, t3, ..., t(nbDelays) given by the sequence.

    """
    # unpack the sequence
    delays, = sequence
    # set successive delays for evolution
    successiveDelays = np.copy(delays)
    successiveDelays[1:] = delays[1:] - delays[:-1]
    if np.any(successiveDelays < 0):
        raise ValueError("The sequence of delays must be increasing")

    # init precap container and compute successive evolutions
    th_pRecap = np.zeros_like(delays, dtype=float)
    for i, t in enumerate(successiveDelays):
        evolution(samples, t, force=None) # free evolution
        th_pRecap[i] = recap_energy(samples, potential, recap_thr)[1]

    return th_pRecap

