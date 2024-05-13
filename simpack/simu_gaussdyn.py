# -*- coding: utf-8 -*-
"""
Functions to run simulations of atomic dynamics in gaussian tweezers, and more
specifically to simulate oscillations and fit the waist of the tweezers.

An experiment of oscillations in the trap destined to measure the waist
consists in releasing the atoms for a first duration T1 to expand the
cloud, then switching back the tweezers for variable delays ti, and
finally releasing them again for a duration T2, to allow the fastest
atoms to escape.


 ON -->|<-- OFF, T1 -->|<--- ON, ti --->|<-- OFF, T2 -->|<-- ON
The sequence is characterized by:
    - T1, the first release duration
    - [t1, t2, t3, ..., tk], the list of oscillation delays,
      with t1 < t2 < t3 < .. < tk.
    - T2, the second release duration

Note that the module implements only the computation of recapture
probability.

The functions are:
    - <trajectory>, to simulate atomic trajectories in a trap-frequency
        measurement sequence
    - <run_simulation>, to run a simulation of a trap-frequency measurement
        sequence


"""

from typing import Callable

import numpy as np

from .simutils import (evolution, recap_energy)
from .config.config import GAUSSDYN_TIMESTEP as dt


# =============================================================================
# Run simulation
# =============================================================================

def trajectory(run: tuple,
               force: tuple[Callable, Callable, Callable],
               isamples: np.ndarray)-> tuple:
    """
    Compute the trajectory of a set of atom samples in a given run of
    a trap oscillation experiment.

    The evolving positions and velocities of the samples are given at each
    time step. Note that this generates a lot of data, hence this function
    should be used for a small number of samples.

    Parameters
    ----------
    run : tuple (T0, T1, tau, T2, T'0)
        Parameters that characterize the experimental run:
        - T0 : float, portion of the trajectory computed before the
               actual experiment
        - T1 : float, 1st release duration
        - tau : float, duration of oscillations in tweezers
        - T2 : float, 2nd release duration
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
    forces = (force, None, force, None, force)
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
                   force: tuple[Callable, Callable, Callable],
                   samples: np.ndarray,
                   recap_thr: float = 0.,
                   substract_mean: bool = False)-> np.ndarray:
    """
    TODO doc
    From a set of samples, generate a simulated curve of recapture
    probability at given delays, for a sequence of oscillations in the
    gauss traps.

    The samples are first left to evolve freely for the first release
    of duration T1.
    Then from the list of delays [t1, t2, ..., tk], where k = nbDelays,
    the samples are left to evolve for nbDelays successive delays
    t1, t2-t1, t3-t2, ..., t(k-1)-tk in the trap potential.
    The recapture probability is computed at each step after a second
    release of duration T2.

    The recapture is determined by the criterion of negative mechanical
    energy in the presence of the trapping potential.

    Parameters
    ----------
    sequence : tuple (T1, delays, T2)
        Parameters that characterize the sequence:
        - T1 : float, 1st release duration
        - delays : np.ndarray, durations [t1, t2, t3, .., tk] with
                   t1 < t2 < t3 < .. < tk.
                   Durations of oscillations in tweezers.
        - T2 : float, 2nd release duration
    potential : callable, 3 arguments (x, y, z)
        Function that returns the trapping potential as a function of
        the position (x, y, z).
    forces : 3-tuple of callable, 3 arguments (x, y, z)
        Functions (Fx, Fy, Fz) that return the components of the force
        as a function of the position (x, y, z).
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
        Simulated recapture probabilities from samples at delays given
        by the sequence.

    """
    # Unpack the sequence
    T1, delays, T2 = sequence
    # set successive delays for evolution
    successiveDelays = np.copy(delays)
    successiveDelays[1:] = delays[1:] - delays[:-1]
    if np.any(successiveDelays < 0):
        raise ValueError("The sequence of delays must be increasing")
    # Subdivise successive delays into atomic timesteps
    steps = []
    for t in successiveDelays:
        N = int(np.ceil(round(t / dt, ndigits=5))) # nb of atomic timesteps
        steps.append(np.full(N, dt, dtype=float))
        try:
            steps[-1][-1] = t - (N-1)*dt
        except IndexError: # edge case t=0
            pass

    # Init precap container
    th_precap = np.zeros_like(delays, dtype=float)
    # First release, for a duration T1
    evolution(samples, T1, force=None)

    # Compute successive evolutions in tweezers
    for i, t in enumerate(steps):
        for step in t:
            evolution(samples, step, force=force)
        # At each delay, compute the second release T2 to get precap
        samples2 = np.copy(samples)
        evolution(samples2, T2, force=None)
        th_precap[i] = recap_energy(samples2, potential, recap_thr)[1]

    return th_precap


