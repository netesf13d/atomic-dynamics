# -*- coding: utf-8 -*-
"""
Functions to run simulations of atomic motion in BoB tweezers, and more
specifically to simulate oscillations.

An experiment of oscillations in BoB traps consists in preparing the atoms
in gaussian tweezers with an offset (x, y, z) with respect to the BoBs,
allowing them to oscillate in the BoBs for a variable duration ti. After
releasing them for a duration T to so that the fastest atoms escape,
they are recaptured in the BoBs for a duration t'i before being recaptured
in gaussian tweezers.
Usually, we have ti + T + t'i constant.

Gauss -->|-dt1-|<-- BoB, ti -->|<--- OFF, T --->|<-- BoB, t'i -->|-dt2-|<-- Gauss
The sequence is characterized by:
    - (x0, y0, z0) the offset of the tweezers' center with respect to the
      BoB's center.
    - dt1, the laser (de)excitation duration.
    - [t1, t2, t3, ..., tk], the 1st list of BoB oscillation delays.
    - T, the release duration.
    - [t'1, t'2, t'3, ..., t'k], the 2nd list of BoB oscillation delays.
    - dt2, the laser de-excitation duration.

Note that the module implements only the computation of recapture
probability.

The functions are:
    - <trajectory>, to simulate atomic trajectories in a trap-frequency
        measurement sequence
    - <run_simulation>, to run a simulation of a trap-frequency measurement
        sequence

TODO
- trajectory
- doc
- everything
- test
"""

from typing import Callable

import numpy as np

from .simutils import (evolution, recap_energy)
from .config.config import BOBDYN_TIMESTEP


# =============================================================================
# Run simulation
# =============================================================================

def trajectory(run: tuple,
               gauss_force: tuple[Callable, Callable, Callable],
               bob_force: tuple[Callable, Callable, Callable],
               isamples: np.ndarray,
               dt: float = BOBDYN_TIMESTEP)-> tuple:
    """
    TODO doc
    Compute the trajectory of a set of atom samples in a given run of
    a trap oscillation experiment.

    The evolving positions and velocities of the samples are given at each
    time step. Note that this generates a lot of data, hence this function
    should be used for a small number of samples.

    Parameters
    ----------
    run : tuple (T0, T1, tau1, T, tau2, T'1, T'0)
        Parameters that characterize the experimental run:
        - T0 : float, portion of the trajectory computed before the
               actual experiment
        - T1 : float, 1st laser excitation release duration
        - tau1 : float, duration of 1st oscillations in tweezers
        - T : float, expansion release duration
        - tau2 : float, duration of 2nd oscillations in tweezers
        - T'1 : float, 2nd laser excitation release duration
        - T'0 : float, portion of the trajectory computed after the
                actual experiment (the recapture)
    gauss_force : tuple (Fx, Fy, Fz) of callable, 3 arguments (x, y, z)
        Functions Fi that return the components of the force excerted by
        gaussian tweezers as a function of the position (x, y, z).
    bob_force : tuple (Fx, Fy, Fz) of callable, 3 arguments (x, y, z)
        Force exerted by the BoB trap potential.
    offset : tuple (x0, y0, z0)
        Offset of the *BoB* potential with respect to the *gaussian*
        potential. The force F_bob is evaluated at (x-x0, y-y0, z-z0).
    isamples : np.ndarray
        Atom samples as an array of shape (6, nbSamples). The format is :
            - ispl[0, i], ispl[1, i], ispl[2, i] :
                x, y, z coordinates of atom i respectively
            - ispl[3, i], ispl[4, i], ispl[5, i] :
                v_x, v_y, v_z components of atom i respectively
    dt : float, optional
        The integration timestep. Defaults to the config value BOBDYN_TIMESTEP.

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
    forces = (gauss_force, None, bob_force, None, bob_force, None, gauss_force)
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
                   gauss_potential: Callable,
                   bob_force: tuple[Callable, Callable, Callable],
                   samples: np.ndarray,
                   recap_thr: float = 0.,
                   substract_mean: bool = False,
                   dt: float = BOBDYN_TIMESTEP)-> np.ndarray:
    """
    From a set of samples, generate a simulated curve of recapture
    probability at given delays, for a sequence of oscillations in the
    BoBs.

    The samples, centered on gaussian tweezers, are first offset to
    `center` them on BoBs.
    Then for each pair of delays (t1, t'1), (t2, t'2), ..., (tk, t'k)
    with k = nbDelays, the samples are evolved:
    - for a duration ti in the BoBs
    - for a duration T in free space
    - for a duration t'i in the BoBs
    Finally, after applying the inverse translation to recenter the
    samples on the gaussian tweezers, the recapture probability is
    computed.

    The recapture is determined by the criterion of negative mechanical
    energy in the presence of the trapping potential.

    NOTE : A spatial offset between the gaussian and Bob must be implemented
    in the potential or the force (I recommend the force)

    Parameters
    ----------
    sequence : tuple (t1, delays, T, t2, Ttot)
        t1 : float, gauss-to-bob switching duration
        delays : np.ndarray, durations [t1, t2, t3, .., tk] with
                 t1 < t2 < t3 < .. < tk.
                 Durations of oscillations in tweezers.
        T : float, main release duration
        t2 : float, bob-to-gauss switching duration
        Ttot : float or np.ndarray, total sequence duration
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
    dt : float, optional
        The integration timestep. Defaults to the config value BOBDYN_TIMESTEP.

    Raises
    ------
    ValueError
        If the sequence delays are not increasing, ie some successive
        delay is negative. Negative time evolution is ill defined.

    Returns
    -------
    th_pRecap : 1D np.ndarray of shape (nbDelays,)
        Simulated recapture probabilities from samples at delays given
        by the sequence.

    """
    # Unpack the sequence
    t1, delays, T, t2, Ttot = sequence

    # Check consistency
    Ttot = np.broadcast_to(Ttot, delays.shape)
    if (idx:=np.nonzero(Ttot < T + t1 + t2 + delays)[0]).size > 0:
        raise ValueError(
            f"total sequence duration too short at delays at indices {idx}")

    successiveDelays = np.copy(delays)
    successiveDelays[1:] -= delays[:-1]
    if np.any(successiveDelays < 0):
        raise ValueError("The sequence of delays must be increasing")
    # Subdivise first delays into atomic timesteps
    steps = []
    for t in successiveDelays:
        N = int(np.ceil(round(t / dt, ndigits=5))) # nb of atomic timesteps
        steps.append(np.full(N, dt, dtype=float))
        try:
            steps[-1][-1] = t - (N-1)*dt
        except IndexError: # edge case t=0
            pass

    # Subdivise second delays into atomic timesteps
    delays2 = Ttot - delays - T - t1 - t2
    steps2 = []
    for t in delays2:
        N = int(np.ceil(t / dt))
        steps2.append(np.full(N, dt, dtype=float))
        try:
            steps2[-1][-1] = t - (N-1)*dt
        except IndexError: # edge case t=0
            pass

    # Init precap container
    th_precap = np.zeros_like(delays, dtype=float)

    # Release during laser excitation/gauss-to-bob switch
    evolution(samples, t1, force=None)

    # Compute atom dynamics
    for i, t in enumerate(steps):
        # First evolution in BoBs
        for step in t:
            evolution(samples, step, force=bob_force)
        samples2 = np.copy(samples)
        # Release for duration T
        evolution(samples2, T, force=None)
        # Second evolution in BoBs
        for step2 in steps2[i]:
            evolution(samples2, step2, force=bob_force)
        # Release during laser de-excitation/bob-to-gauss switch
        evolution(samples2, t2, force=None)
        th_precap[i] = recap_energy(samples2, gauss_potential)[1]

    if substract_mean:
        th_precap -= np.mean(th_precap)

    return th_precap

