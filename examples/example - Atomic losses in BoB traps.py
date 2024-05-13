# -*- coding: utf-8 -*-
"""
An example to show what is possible with the simpack.

This script computes atomic losses in BoB traps as a function of the atomic
temperature. The atoms are transferred from a thermal equilibrium in 1 mK-deep
gaussian beam dipole traps to much shallower (70 uK) BoB traps, resulting in
loss of hot atoms. See Appendix D.2 of my thesis for details.
https://theses.hal.science/tel-04551702

The script is divided in three parts.
* Setup the potential and forces from the BoB potential computed in
  `example - BoB potential computation`.
* Compute atomic trajectories at different temperatures.
  - Initial atomic positions and velocities are sampled from the equilibrium
    distribution in a gaussian beam trap
  - Atomic dynamics in BoB traps is computed, recording the loss time of atoms.
  - Compute the recapture probability and the loss times distribution...
* Plot and save data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from scipy.special import erf

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


########## Recapture model ##########

def precap_model(x: np.ndarray, thr_temperature: float)-> np.ndarray:
    """
    Exponential decay with offset y0.
    f(x) = y0 + A * exp(- x / tau)
    """
    u = thr_temperature/x
    return erf(np.sqrt(u)) - np.sqrt(4*u/np.pi) * np.exp(- u)


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
# Compute ponderomotive force field
# =============================================================================

########## Load pre-computed potentials ##########

### Raw BoB potential
bob = sim.Potential_Array(source=POT_PATH + "BoB_pot.npz")
## Trapping threshold
thr, _ = bob.get_threshold()
thr_Hz = thr / h # trap threshold in Hz
thr_K = thr / k_B # K
## Light shift at the center of the trap
_, Emin = bob.get_interior()
Emin_Hz = Emin / h # Hz
Emin_K = Emin / k_B # K

## threshold trapping temperature
Tthr = (thr_K - Emin_K) * beta_ponder * P0 # K


########## Ponderomotive force and potential energy ##########

## Ponderomotive force from the 52C potential, compute by discrete differentiation
force = sim.build_force(bob.dgrid, offset=offset, amplitude=beta_ponder*P0)

## Trapping condition
rlim = 1.4e-6 # limit radius
zlim = 4e-6 # propagation axis limit
def trapped(x, y, z)-> bool:
    """
    True if an atom at position (x, y, z) is considered trapped.
    Conservative condition to be sure of atomic losses.
    """
    r2 = x**2 + y**2
    return np.logical_or(r2 < rlim**2, z**2 < zlim**2)


# =============================================================================
# ########## Atomic trajectory simulations ##########
# =============================================================================

########## Simulation parameters ##########
## For each temperature, carry `repetitions` repetitions of the following.
## The trajectories of `n` atoms are computed over `nb_steps` steps, with
## `timestep` time step, and the loss time (when losses occur) of atoms is
## recorded at each step.

# atomic temperatures
temperatures = np.linspace(1e-6, 99e-6, 11, endpoint=True) # K
# number of repetitions of the computation
repetitions = 3
# samples per repetition
n = 5000
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
mean_tloss = np.zeros_like(temperatures, dtype=float) # s, mean shift
stdev_tloss = np.zeros_like(temperatures, dtype=float) # s, stdev of shift
bins = np.linspace(-1., 399., (nbbin:=200)+1, endpoint=True) # s, histogram bins
hist = np.zeros((len(temperatures), nbbin), dtype=float) # histogram

## Recapture probability
precap = np.zeros_like(temperatures, dtype=float) # recaptured fraction at t_tot

## Coherence
coherence = np.zeros((len(temperatures), nb_steps+1), dtype=float) # coherence

for i, T in enumerate(temperatures):
    print(f"\nComputing trajectories at temperature {T*1e6:.2f} uK")
    ## Run simulation
    filt = np.zeros((N,), dtype=bool)
    raw_tloss = np.zeros((N,), dtype=int)

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

        # True for trapped atoms
        filt[j*n:(j+1)*n] = trapped(x[-1], y[-1], z[-1])
        raw_tloss[j*n:(j+1)*n] = np.argmin(trapped(x, y, z), axis=0)

    ### Fraction of recaptured atoms
    precap[i] = np.mean(filt)

    ### Loss time conditioned to the recapture
    tloss = raw_tloss[np.logical_not(filt)] * timestep * 1e6 # us
    mean_tloss[i] = np.mean(tloss)
    stdev_tloss[i] = np.std(tloss)
    hist[i] = np.histogram(tloss, bins=bins, density=False)[0]
    print(f"Average loss time at T = {T*1e6:.3} uK, trap power {P0*1e3} mW: "
          f"{mean_tloss[i]} +- {stdev_tloss[i]} us")


# =============================================================================
# ########## Plot ##########
# =============================================================================
Tmax = np.max(temperatures)*1e6
### Build custom colormap
vmin, vmax = 0., 2.
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = colormaps.get_cmap("plasma")
cmap = colors.ListedColormap(cmap(np.linspace(0.0, 1., 256)))

### Initialize the figure
fig, ax = plt.subplots(1, 2, figsize=(9, 4),
                        squeeze=False, dpi=200)
fig.subplots_adjust(left=0.055, right=0.98,
                    bottom=0.12, top=0.92,
                    hspace=0.25, wspace=0.34)

########## Recapture vs T ##########
ax[0, 0].set_xlim((0, Tmax))
ax[0, 0].set_ylim(0., 1.)
ax[0, 0].tick_params(
    axis='both', direction='out', labelsize=9, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax[0, 0].set_xticks(np.linspace(0., 100., 6))
ax[0, 0].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
ax[0, 0].grid(which='major', alpha=0.5)
ax[0, 0].grid(which='minor', alpha=0.2)

ax[0, 0].plot(temperatures*1e6, precap,
              marker="o", ls="")
ax[0, 0].plot(temp:=np.linspace(1., 100., 100), precap_model(temp*1e-6, Tthr),
              marker="", ls="-")

ax[0, 0].set_xlabel(r"Temperature (uK)", fontsize=11, labelpad=3)
ax[0, 0].set_ylabel(r"Recapture prob.", fontsize=11, labelpad=3)


########## Escape time vs T ##########
ax[0, 1].set_xlim((0, Tmax))
ax[0, 1].set_ylim(0., 400.)
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
    temperatures*1e6, mean_tloss,
    marker='o', ls="-", color="0.5")

bbins = (bins[1:] + bins[:-1])/2
X, Y = np.meshgrid(temperatures*1e6, bbins)
ax[0, 1].pcolormesh(X, Y, hist.transpose(), shading="gouraud",
                    vmin=0., vmax=250., cmap="inferno",
                    edgecolors="none", lw=0.)

ax[0, 1].set_xlabel(r"Temperature (uK)", fontsize=11, labelpad=3)
ax[0, 1].set_ylabel(r"Escape time (us)", fontsize=11, labelpad=5)


# =============================================================================
# ########## Save data ##########
# =============================================================================
fpath_out = "./data/"

data_out = {
    'trap_power': np.array(P0), # W
    'trap_depth_Hz': np.array((thr_Hz - Emin_Hz) * beta_ponder * P0 ),
    'trap_depth_K': np.array((thr_K - Emin_K) * beta_ponder * P0),

    'nb_samples': N,
    'temperatures': temperatures, # K
    'bins': bins, # Hz
    'tloss_histogram': hist,

    'precap': precap,

    'mean_tloss': mean_tloss, # us
    'stdev_tloss': stdev_tloss, # us
    }

# np.savez_compressed(fpath_out + "BoB_losses.npz", **data_out)
fig.savefig(fpath_out + "./summary_BoB_losses.png")
