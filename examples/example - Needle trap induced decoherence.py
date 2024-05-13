# -*- coding: utf-8 -*-
"""
An example to show what is possible with the simpack.

This script computes the coherence loss of a circular states superposition
(|70C> + |72C>) / sqrt(2) in a given trapping potential. The coherence loss is
due to the small differential of ponderomotive energy coupled to the atomic
motion in the trap. See Appendix D.2 of my thesis for details.
https://theses.hal.science/tel-02903456

Here we consider the trapping potential described in the Conclusion and
Appendix C.2 of Rodrigo Cortinas' thesis.
https://theses.hal.science/tel-02903456

The script is divided in four parts.
* Setup the expression of the potential (analytical in this case).
* Compute the ponderomotive energy for both states |70C> and |72C> on a
  discrete grid, along with the ponderomotive force to compute the classical
  dynamics.
* Compute trajectories for atomic ensembles sampled at thermal equilibrium with
  different temperatures. Record the energy difference between |70C> and |72C>
  along the trajectory to compute the coherence loss.
* Plot and save data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps, cm

# simpack in parent directory
pdir = Path(__file__).resolve().parents[1]
if not (simd := str(pdir)) in sys.path:
    sys.path.append(simd)
import simpack as sim


from scipy.constants import physical_constants
hbar = physical_constants['reduced Planck constant'][0]
h = physical_constants['Planck constant'][0]
k_B = physical_constants['Boltzmann constant'][0]
u = physical_constants['atomic mass constant'][0]
m_Rb87 = 86.909180520*u


# custom trajectory computation with sub-sampling
def trajectory(t_eval: list[float],
               force: tuple[callable],
               isamples: np.ndarray,
               dt: float = 1e-6)-> np.ndarray:
    """
    Compute a trajectory for the given atomic samples, with position/velocity
    sampled at given times.

    Parameters
    ----------
    t_eval : list[float], len N
        The times at which the trajectory points are returned. Must be
        a sequence of increasing numbers.
    force : tuple (Fx, Fy, Fz) of callable, 3 arguments (x, y, z)
        Force exerted by the trap potential.
    isamples : np.ndarray, shape (6, nbSamples)
        Atom samples. The format is :
            - ispl[0, i], ispl[1, i], ispl[2, i] :
                x, y, z coordinates of atom i respectively
            - ispl[3, i], ispl[4, i], ispl[5, i] :
                v_x, v_y, v_z components of atom i respectively

    Returns
    -------
    traj : np.ndarray, shape (N, 6, nbSamples)
        Atomic samples positions and velocity at each time.
        traj_spl[i, :, :] = trajectory samples at time[i]
        (same format as isamples)
    """
    # Compute successive delays between evaluation times
    delays = np.copy(t_eval)
    delays[1:] -= t_eval[:-1]
    if np.any(delays < 0):
        raise ValueError("The sequence of delays must be increasing")
    # Subdivise delays into atomic timesteps
    steps = []
    for t in delays:
        n = int(np.ceil(round(t / dt, ndigits=5))) # nb of atomic timesteps
        steps.append(np.full(n, dt, dtype=float))
        try:
            steps[-1][-1] = t - (n-1)*dt
        except IndexError: # edge case t=0
            pass
    # Compute trajectory
    traj = np.empty((len(t_eval), *isamples.shape), dtype=float)
    temp = isamples
    for k, t in enumerate(steps):
        traj[k, :, :] = np.copy(temp)
        for step in t: # inplace modification of the position/velocity
            sim.evolution(traj[k, :, :], step, force=force)
        temp = traj[k, :, :]
    return traj


# =============================================================================
# Configuration and physical parameters
# =============================================================================

POT_PATH = "./potentials/"


lambda_trap = 300. * 1e-9 # m, trap wavelength
NA = 0.5 # numerical aperture
P0 = 0.05 # W, beam power

# Deduced parameters
beta_ponder = sim.ponderomotive_coef(lambda_trap) # Ponderomotive strength
w0 = lambda_trap / (2*NA)
zR = np.pi * w0**2 / lambda_trap


def potential(x, y, z):
    """
    Numerical computation of the potential.
    """
    w2 = w0**2 * (1 + (z / zR)**2)
    exp = np.exp(-2 * (x**2+y**2) / w2)
    return beta_ponder * 2/np.pi * P0 / w2 * exp


# =============================================================================
# Compute ponderomotive energy and force
# =============================================================================
### Sample the potential on a meshgrid
# evaluation domain in meters
ext = np.array([[-3. * w0, 3. * w0],
                [-3. * w0, 3. * w0],
                [-1.5 * zR, 1.5 * zR]])
# grid pixel size
pxsz = np.array([20e-9, 20e-9, 40e-9])
# grid index of the trap center
loc = (ext[:, 1] / pxsz).astype(int)
# shape of the meshgrid
sh = 2 * loc + 1
# sampled potential
raw_pot = sim.Potential_Array(source=potential,
                              shape=sh,
                              pxsize=pxsz,
                              loc=loc)
raw_pot.plot_section()


### Compute the convolution
# quantization axisof the circular state
axis = 2
# pixel size of the circular electron probability distribution
pd_pxsz = (10e-9, 10e-9, 5e-9)
# grid pixel size of the convolved potential
pot_pxsz = (20e-9, 20e-9, 40e-9)

## |70C> potential
pd70 = sim.Circ_Prob_Density(n=70, norm=1.)
pd70 = pd70.prob_density_array(pd_pxsz,
                                  eval_mode="analytical",
                                  optimize_shape=True)
pd70 = sim.Data_Grid(**pd70)
pot70 = raw_pot.compute_potential(pxsize=pot_pxsz,
                                  probability_density=pd70,
                                  axis=axis,
                                  symmetries=(0, 1, 2))
## Characterisation of the potential: trap depth, curvature, ...
pot70 = sim.Potential_Array(source=pot70)
# pot70.plot_section() # What does it look like ?
# Analyze trapping threshold
pot70.analyze("threshold")
pot70_thr, pot70_thr_err = pot70.get_threshold()
pot70_thr_Hz = pot70_thr / h # trap threshold in Hz
pot70_thr_K = pot70_thr / k_B # K
# Get the trap interior
pot70.analyze("interior")
pot70_in, pot70_Emin = pot70.get_interior()
pot70_Emin_Hz = pot70_Emin / h # Light-shift at the center of the trap in Hz
pot70_Emin_K = pot70_Emin / k_B # K
# Determine curvature, and trap frequencies
pot70.analyze("curvature", extent=[1.6*w0, 1.6*w0, 2*zR])
pot70_trap_curv, *_ = pot70.get_curvature()
pot70_trap_freq = np.sqrt(2 * pot70_trap_curv / m_Rb87) / (2*np.pi) # Hz
# save...
# pot70.save(POT_PATH + "Needle_pot_70C.npz")
# ...and load the data
# pot70 = sim.Potential_Array(source=POT_PATH + "Needle_pot_70C.npz")

## |72C> potential
pd72 = sim.Circ_Prob_Density(n=72, norm=1.)
pd72 = pd72.prob_density_array(pd_pxsz,
                                  eval_mode="analytical",
                                  optimize_shape=True)
pd72 = sim.Data_Grid(**pd72)
pot72 = raw_pot.compute_potential(pxsize=pot_pxsz,
                                  probability_density=pd72,
                                  axis=axis,
                                  symmetries=(0, 1, 2))
## Characterisation of the potential: trap depth, curvature, ...
pot72 = sim.Potential_Array(source=pot72)
pot72.plot_section() # What does it look like ?
# Analyze trapping threshold
pot72.analyze("threshold")
pot72_thr, pot72_thr_err = pot72.get_threshold()
pot72_thr_Hz = pot72_thr / h # trap threshold in Hz
pot72_thr_K = pot72_thr / k_B # K
# Get the trap interior
pot72.analyze("interior")
pot72_in, pot72_Emin = pot72.get_interior()
pot72_Emin_Hz = pot72_Emin / h # Light-shift at the center of the trap in Hz
pot72_Emin_K = pot72_Emin / k_B # K
# Determine curvature, and trap frequencies
pot72.analyze("curvature", extent=[1.6*w0, 1.6*w0, 2*zR])
pot72_trap_curv, *_ = pot72.get_curvature()
pot72_trap_freq = np.sqrt(2 * pot72_trap_curv / m_Rb87) / (2*np.pi) # Hz
# save...
# pot72.save(POT_PATH + "Needle_pot_72C.npz")
# ...and load the data
# pot72 = sim.Potential_Array(source=POT_PATH + "Needle_pot_72C.npz")


### Compute the ponderomotive force and potential energy
# Ponderomotive force from the 72C potential
F = sim.build_force(pot72.dgrid) # compute by discrete differentiation

# Ponderomotive potential energy for the states |70C> and |72C>
V72 = sim.build_potential(pot72.dgrid)
V70 = sim.build_potential(pot70.dgrid)

# Trapping condition
interior72 = sim.Data_Grid(pot72_in, pot72.dgrid.pxsize, pot72.dgrid.loc)
trapped = sim.build_potential(interior72, method='nearest')


# =============================================================================
# Coherence and lightshift simulation
# =============================================================================

########## Simulation parameters ##########
## For each temperature, carry `repetitions` repetitions of the following.
## The trajectories of `n` atoms are computed over a time `t_tot`, and the
## energy (lightshift) difference is recorded at times `times`.

# atomic temperatures
temperatures = np.logspace(-7, np.log10(30e-6), 16, endpoint=True) # K
# number of repetitions of the computation
repetitions = 1
# samples per repetition
n = 2000
# total number of samples
N = n*repetitions
# sampling times
times = np.linspace(0., 5e-3, (k:=501), endpoint=True)
# total simulation time
t_tot = times[-1]
# sampling timestep
tstep = times[1] - times[0]


########## Classical dynamics simulation ##########
mean_df = np.zeros_like(temperatures, dtype=float) # frequency shift
stdev_df = np.zeros_like(temperatures, dtype=float) # stdev of frequency shift

bins = (pot72_Emin_Hz - pot70_Emin_Hz) \
       + np.linspace(-50e3, 200e3, (nbbin:=250)+1, endpoint=True)
hist = np.zeros((len(temperatures), nbbin), dtype=float)

ploss = np.zeros((len(temperatures),), dtype=float) # atoms lost at t_tot
coherence = np.zeros((len(temperatures), k), dtype=float) # coherence

for i, T in enumerate(temperatures):
    print(f"\nComputing trajectories at temperature {T*1e6:.2f} uK")
    ## Run simulation
    domega = np.zeros((N, k), dtype=float)
    filt = np.zeros((N,), dtype=bool)

    for j in range(repetitions):
        # initial potition and velocity
        spl = sim.harmonic_sampling(n, T, *(pot72_trap_freq * 2*np.pi))

        # trajectory
        ts0 = time.time_ns()/10e8
        traj = trajectory(times, F, spl)
        ts1 = time.time_ns()/10e8
        print(f"Atomic trajectories for {n} samples ({len(times)} "
              f"timesteps) in {ts1-ts0:.2f} s")
        x, y, z = traj[:, 0, :], traj[:, 1, :], traj[:, 2, :]

        # ponderomotive energies along the trajectory
        E_72 = V72(x, y, z)
        E_70 = V70(x, y, z)
        domega[j*n:(j+1)*n, :] = np.transpose(E_72 - E_70) / hbar

        # True for trapped atoms
        filt[j*n:(j+1)*n] = trapped(x[-1], y[-1], z[-1])

    filt_domega = domega[filt, :]

    ## Loss probability
    ploss[i] = 1. - np.mean(filt)
    ## Average lightshift conditioned to the recapture of atoms
    mean_df[i] = np.mean(filt_domega)/(2*np.pi)
    stdev_df[i] = np.std(filt_domega)/(2*np.pi)
    hist[i] = np.histogram(filt_domega/(2*np.pi), bins=bins, density=True)[0]
    print(f"Average shift at T = {T*1e6:.3f} uK: "
          f"{mean_df[i]/1e3:.3f} +- {stdev_df[i]/1e3:.3f} kHz")

    # Dephasing
    phi = tstep * np.cumsum(filt_domega, axis=1)
    coherence[i, :] = np.abs(np.sum(np.exp(1J * phi), axis=0) / np.sum(filt))

# Coherence time
T2 = np.argmin(coherence > 1/2, axis=1) * tstep


# =============================================================================
# ########## Plot ##########
# =============================================================================

Tmax = np.max(temperatures)*1e6
### Build custom colormap
vmin, vmax = 0., np.log10(Tmax)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = colormaps.get_cmap("plasma")
cmap = colors.ListedColormap(cmap(np.linspace(0.0, 1., 256)))

### Initialize the figure
fig, ax = plt.subplots(2, 2, figsize=(9, 7),
                        squeeze=False, dpi=200)
fig.subplots_adjust(left=0.055, right=0.98,
                    bottom=0.06, top=0.92,
                    hspace=0.25, wspace=0.34)
for a in np.nditer(ax, flags=['refs_ok']):
    a.flat[0].grid(which='major', alpha=0.5)
    a.flat[0].grid(which='minor', alpha=0.2)

########## Loss vs T ##########
ax[0, 0].set_xlim((0, Tmax))
ax[0, 0].set_ylim(0., 1.)
ax[0, 0].tick_params(
    axis='both', direction='out', labelsize=9, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[0, 0].set_xticks(np.linspace(0., Tmax, 6))
ax[0, 0].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])

ax[0, 0].plot(temperatures*1e6, ploss,
              marker='o', ls="-")

ax[0, 0].set_xlabel(r"Temperature (uK)", fontsize=11, labelpad=3)
ax[0, 0].set_ylabel(r"Loss probability", fontsize=11, labelpad=3)


########## Frequency shift vs T ##########
ax[0, 1].set_xlim((0, Tmax))
ax[0, 1].set_ylim(bins[0]/1e3, bins[-1]/1e3)
ax[0, 1].tick_params(
    axis='both', direction='out', labelsize=9, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[0, 1].set_xticks(np.linspace(0., Tmax, 6))
ax[0, 1].grid(False)

ax[0, 1].plot(
    temperatures*1e6, mean_df/1e3,
    marker='o', ls="-", color="0.8")
ax[0, 1].plot(
    ax[0, 1].get_xlim(), 2*[(V72(0, 0, 0) - V70(0, 0, 0)) / h / 1e3],
    marker='', ls="--", color="0.7", lw=0.8)

bbins = (bins[1:] + bins[:-1])/2
X, Y = np.meshgrid(temperatures*1e6, bbins/1e3)
ax[0, 1].pcolormesh(X, Y, hist.transpose()*4e4, shading="gouraud",
                    vmin=0., vmax=1., cmap="inferno",
                    edgecolors="none", lw=0.)

ax[0, 1].set_xlabel(r"Temperature (uK)", fontsize=11, labelpad=3)
ax[0, 1].set_ylabel(r"Frequency shift (kHz)", fontsize=11, labelpad=5)


########## coherence vs time ##########
ax[1, 0].set_xlim(0, t_tot*1e3)
ax[1, 0].set_ylim(0, 1)

ax[1, 0].tick_params(
    axis='both', direction='out',
    labelsize=10, pad=1.5,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[1, 0].set_xticks(np.linspace(0., t_tot*1e3, 5))
# ax[1, 0].set_xticks([25, 75, 125], minor=True)
ax[1, 0].set_yticks(np.linspace(0., 1., 6))

for i, T in enumerate(temperatures):
    ax[1, 0].plot(times*1e3, coherence[i],
                  marker="", ls="-", color=cmap(np.log10(1e6*T)/2), lw=1.)

ax[1, 0].set_xlabel(r"Time (ms)", fontsize=11, labelpad=2)
ax[1, 0].set_ylabel(r"Coherence", fontsize=11, labelpad=4)


### Colorbar
x0, y0, dx, dy = ax[1, 0].get_position().bounds
x1 = x0 + dx
cax1 = fig.add_axes(rect=(x1+0.01, y0, 0.02, dy))
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1,
                  orientation="vertical",
                  ticklocation="right", alpha=1.)
cax1.tick_params(axis="y", which="major", direction="in",
                 size=3.,
                 color="k", labelsize=9, pad=2)
cax1.tick_params(axis="y", which="minor", direction="in",
                 size=2.,
                 color="k", labelsize=9, pad=1)

cax1.set_yticks([0., 1., np.log10(Tmax)],
                [r"$1$", r"$10$", f"${int(Tmax)}$"], minor=False)
ticks = [np.log10(i)
         for i in [2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90]
         if i < Tmax]
cax1.set_ylabel(r"Temperature (uK)",
                ha="center", va="center", rotation=-90, labelpad=2, fontsize=11)


########## coherence vs T ##########
ax[1, 1].set_xlim(0, Tmax)
ax[1, 1].set_ylim(0., 0.5)
ax[1, 1].tick_params(
    axis='both', direction='out', labelsize=9, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[1, 1].set_xticks(np.linspace(0., Tmax, 6))

ax[1, 1].plot(
    temperatures*1e6, T2*1e3,
    marker='o', ls="-")

ax[1, 1].set_xlabel(r"Temperature (uK)", fontsize=11, labelpad=3)
ax[1, 1].set_ylabel(r"T2 (ms)", fontsize=11, labelpad=3)


# =============================================================================
# ########## Save data ##########
# =============================================================================
fpath_out = "./data/"

data_out = {
    'trap_freq': np.array(pot72_trap_freq), # Hz
    'trap_thr_Hz': np.array(pot72_thr_Hz), # Hz
    'trap_thr_K': np.array(pot72_thr_K), # K
    'trap_Emin_Hz': np.array(pot72_Emin_Hz), # Hz
    'trap_Emin_K': np.array(pot72_Emin_K), # K
    'trap_depth_Hz': np.array(pot72_thr_Hz - pot72_Emin_Hz), # Hz
    'trap_depth_K': np.array(pot72_thr_K - pot72_Emin_K), # K
    'min_shift': np.array(pot72_Emin_Hz - pot70_Emin_Hz), # Hz

    'temperatures': temperatures, # K
    'bins': bins, # Hz
    'f_histogram': hist,

    'freq_shift': mean_df, # Hz
    'stdev_freq_shift': stdev_df, # Hz

    'times_ms': times * 1e3,
    'coherence': coherence,
    'T2': T2,
    }

# np.savez_compressed(fpath_out + NeedleTrap_shift_and_decoherence.npz", **data_out)
fig.savefig(fpath_out + "NeedleTrap_summary_shift_and_decoherence.png")


