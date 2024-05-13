# -*- coding: utf-8 -*-
"""
An example to show what is possible with the simpack.

This script computes the coherence loss of a circular states superposition
(|50C> + |52C>) / sqrt(2) in a given trapping potential. The coherence loss is
due to the small differential of ponderomotive energy coupled to the atomic
motion in the trap. See Appendix D.2 of my thesis for details.
https://theses.hal.science/tel-04551702

Here we consider the bottle beam trapping potential discussed extensively in my
thesis.

The script is divided in three parts.
* Setup the potential and forces from the BoB potential computed in
  `example - BoB potential computation`.
* Compute atomic trajectories at different temperatures.
  - Initial atomic positions and velocities are sampled from the equilibrium
    distribution in a gaussian beam trap
  - Atomic dynamics in BoB traps is computed, recording the energy difference
    between |50C> and |52C> at regular intervals
  - Compute various quantities: coherence vs time, loss probability, light
    shift distribution...
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

# light-shift coefficient for F=1,2 level
LScoef_5S = -0.01816429616015425 # Hz/(W/m^2)
# light-shift coefficient for the F=1 to F'=2 repumper transistion
LScoef_5S5P = 0.023876855585657605 # Hz/(W/m^2)
beta = - LScoef_5S / LScoef_5S5P * h # J/(W/m^2)


# =============================================================================
# Configuration and physical parameters
# =============================================================================

POT_PATH = "./potentials/"

## Gaussian beam parameters
gauss_trap_wavelength = 821. * 1e-9 # m, trap wavelength
w0 = np.array([1.2 * 1e-6, 1.2 * 1e-6])  # m, waist
LS = 27.1 * 1e6 # Hz, trap depth
# deduced parameters
z_R = np.pi * w0**2 / gauss_trap_wavelength # m
V0_gauss = beta * LS # potential amplitude
gparam = {'V0': V0_gauss, 'waist': w0, 'z_R': z_R, 'dz': 0.}

## BoB parameters
bob_trap_wavelength = 821. * 1e-9 # m, trap wavelength
P0 = 0.02 # W, beam power
offset = (0., 0., 0.) # m, offset with respect to the gaussian beam
# Deduced parameters
beta_ponder = sim.ponderomotive_coef(bob_trap_wavelength) # Ponderomotive strength


# =============================================================================
# Compute ponderomotive energy and force
# =============================================================================

########## Load pre-computed potentials ##########

# bob = sim.Potential_Array(source=POT_PATH + "BoB_pot.npz")

### |50C> potential
pot50 = sim.Potential_Array(source=POT_PATH + "BoB_pot_50C_x.npz")
## Trapping threshold
pot50_thr, _ = pot50.get_threshold()
pot50_thr_Hz = pot50_thr / h # trap threshold in Hz
pot50_thr_K = pot50_thr / k_B # K
## Light shift at the center of the trap
_, pot50_Emin = pot50.get_interior()
pot50_Emin_Hz = pot50_Emin / h # Hz
pot50_Emin_K = pot50_Emin / k_B # K
## Trap frequency
pot50_trap_curv, *_ = pot50.get_curvature()
pot50_trap_freq = np.sqrt(2 * pot50_trap_curv / m_Rb87) / (2*np.pi) # Hz

### |52C> potential
pot52 = sim.Potential_Array(source=POT_PATH + "BoB_pot_52C_x.npz")
## Trapping threshold
pot52_thr, _ = pot52.get_threshold()
pot52_thr_Hz = pot52_thr / h # trap threshold in Hz
pot52_thr_K = pot52_thr / k_B # K
## Light shift at the center of the trap
_, pot52_Emin = pot52.get_interior()
pot52_Emin_Hz = pot52_Emin / h # Hz
pot52_Emin_K = pot52_Emin / k_B # K
## Trap frequency
pot52_trap_curv, *_ = pot52.get_curvature()
pot52_trap_freq = np.sqrt(2 * pot52_trap_curv / m_Rb87) / (2*np.pi) # Hz


########## Ponderomotive force and potential energy ##########

## Ponderomotive force from the 52C potential, compute by discrete differentiation
force = sim.build_force(pot52.dgrid, offset=offset, amplitude=beta_ponder*P0)

## Ponderomotive potential energy for the states |50C> and |52C>
V52 = sim.build_potential(pot52.dgrid, offset=offset, amplitude=beta_ponder*P0)
V50 = sim.build_potential(pot50.dgrid, offset=offset, amplitude=beta_ponder*P0)

## Trapping condition
rlim = pot52.dgrid.extent[0] / 2 # limit radius
zlim = pot52.dgrid.extent[2] / 2 # propagation axis limit
def trapped(x, y, z)-> bool:
    """
    True if an atom at position (x, y, z) is considered trapped.
    Conservative condition to be sure of atomic losses.
    """
    r2 = x**2 + y**2
    return np.logical_or(r2 < rlim**2, z**2 < zlim**2)

# =============================================================================
# Coherence and lightshift simulation
# =============================================================================

########## Simulation parameters ##########
## For each temperature, carry `repetitions` repetitions of the following.
## The trajectories of `n` atoms are computed over `nb_steps` steps, with
## `timestep` time step, and the energy (lightshift) difference is recorded at
## each step`.

# atomic temperatures
temperatures = np.logspace(-6, np.log10(50e-6), 16, endpoint=True) # K
# number of repetitions of the computation
repetitions = 1
# samples per repetition
n = 2000
# sampling times
timestep = 2e-6
# sampling timestep
nb_steps = 3000

# total number of samples
N = n*repetitions
# total simulation time
t_tot = timestep*nb_steps
# experimental sequence
seq = (0., 0., t_tot, 0., 0., 0., 0.)


########## Classical dynamics simulation ##########

## Light shift statistics
mean_df = np.zeros_like(temperatures, dtype=float) # Hz, mean shift
stdev_df = np.zeros_like(temperatures, dtype=float) # Hz, stdev of shift
bins = np.linspace(0., 4000., (nbbin:=400)+1, endpoint=True) # Hz, histogram bins
hist = np.zeros((len(temperatures), nbbin), dtype=float) # histogram

## Loss probability
ploss = np.zeros((len(temperatures),), dtype=float) # atoms lost at t_tot

## Coherence
coherence = np.zeros((len(temperatures), nb_steps+1), dtype=float) # coherence

for i, T in enumerate(temperatures):
    print(f"\nComputing trajectories at temperature {T*1e6:.2f} uK")
    ## Run simulation
    domega = np.zeros((N, nb_steps+1), dtype=float)
    filt = np.zeros((N,), dtype=bool)

    for j in range(repetitions):
        # initial potition and velocity by rejection sampling
        spl = sim.gbeam_sampling(n, T, **gparam) # higher precision

        # trajectory
        ts0 = time.time_ns()/10e8
        times, traj = sim.traj_bobdyn(seq, force, force, spl, dt=timestep)
        ts1 = time.time_ns()/10e8
        print(f"Atomic trajectories for {n} samples ({len(times)} "
              f"timesteps) in {ts1-ts0:.2f} s")
        x, y, z = traj[:, 0, :], traj[:, 1, :], traj[:, 2, :]

        # ponderomotive energies along the trajectory
        E_52 = V52(x, y, z)
        E_50 = V50(x, y, z)
        domega[j*n:(j+1)*n, :] = np.transpose(E_52 - E_50) / hbar

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
    phi = timestep * np.cumsum(filt_domega, axis=1)
    coherence[i, :] = np.abs(np.sum(np.exp(1J * phi), axis=0) / np.sum(filt))

# Coherence time
T2 = np.argmin(coherence > 1/2, axis=1) * timestep


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
ax[0, 1].set_ylim(0.8, 3.2)
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
    ax[0, 1].get_xlim(), 2*[(V52(0, 0, 0) - V50(0, 0, 0)) / h / 1e3],
    marker='', ls="--", color="0.7", lw=0.8)

bbins = (bins[1:] + bins[:-1])/2
X, Y = np.meshgrid(temperatures*1e6, bbins/1e3)
ax[0, 1].pcolormesh(X, Y, hist.transpose()*1e3, shading="gouraud",
                    vmin=0., vmax=2., cmap="inferno",
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
ax[1, 1].set_ylim(0., 2.)
ax[1, 1].tick_params(
    axis='both', direction='out', labelsize=9, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[1, 1].set_xticks(np.linspace(0., Tmax, 6))
ax[1, 1].set_yticks([0., 0.5, 1., 1.5, 2.])

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
    'trap_freq': np.array(pot52_trap_freq), # Hz
    'trap_thr_Hz': np.array(pot52_thr_Hz), # Hz
    'trap_thr_K': np.array(pot52_thr_K), # K
    'trap_Emin_Hz': np.array(pot52_Emin_Hz), # Hz
    'trap_Emin_K': np.array(pot52_Emin_K), # K
    'trap_depth_Hz': np.array(pot52_thr_Hz - pot52_Emin_Hz), # Hz
    'trap_depth_K': np.array(pot52_thr_K - pot52_Emin_K), # K
    'min_shift': np.array(pot52_Emin_Hz - pot50_Emin_Hz), # Hz

    'temperatures': temperatures, # K
    'bins': bins, # Hz
    'f_histogram': hist,

    'freq_shift': mean_df, # Hz
    'stdev_freq_shift': stdev_df, # Hz

    'times_ms': times * 1e3,
    'coherence': coherence,
    'T2': T2,
    }

# np.savez_compressed(fpath_out + BoBtrap_shift_and_decoherence.npz", **data_out)
fig.savefig(fpath_out + "BoBtrap_summary_shift_and_decoherence.png")


