# -*- coding: utf-8 -*-
"""
An example to show what is possible with the simpack.

This script computes and plots some atomic trajectories in BoB traps.
The atoms are transferred from a thermal equilibrium in 1 mK-deep
gaussian beam dipole traps to much shallower (70 uK) BoB traps, resulting in
an out-of-equilibrium dynamics. See Appendix D.2 of my thesis for details.
https://theses.hal.science/tel-04551702

The script is divided in three parts.
* Setup the potential and forces from the BoB potential computed in
  `example - BoB potential computation`.
* Compute atomic trajectories at a given temperature.
  - Initial atomic positions and velocities are sampled from the equilibrium
    distribution in a gaussian beam trap
  - Atomic dynamics in BoB traps is computed, recording the full trajectory of
    atoms, and the energy along the trajectory.
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

## Sampler for initial atomic positions
sampler = sim.normal_gbeam_sampler(**gparam)

# =============================================================================
# Compute ponderomotive energy and force
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

## Ponderomotive potential energy for the states |50C> and |52C>
pot = sim.build_potential(bob.dgrid, offset=offset, amplitude=beta_ponder*P0)

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
temperature = 14.1e-6 # K
# number of repetitions of the computation
repetitions = 1
# samples per repetition
n = 5000
# sampling times
timestep = 2e-7
# trajectory computation timestep
nb_steps = 2000

# total number of samples
N = n*repetitions
# total simulation time
t_tot = timestep*nb_steps
# experimental sequence
seq = (0., 0., t_tot, 0., 0., 0., 0.)


########## Classical dynamics simulation ##########

Ec = np.zeros((nb_steps+1, N), dtype=float)
Ep = np.zeros((nb_steps+1, N), dtype=float)
filt = np.zeros((N,), dtype=bool)
for j in range(repetitions):
    spl = sampler(n, temperature) # harmonic approx sampling

    # trajectory
    ts0 = time.time_ns()/10e8
    times, traj = sim.traj_bobdyn(seq, force, force, spl, dt=timestep)
    ts1 = time.time_ns()/10e8
    print(f"Atomic trajectories for {n} samples ({len(times)} "
          f"timesteps) in {ts1-ts0:.2f} s")

    x, y, z = traj[:, 0, :], traj[:, 1, :], traj[:, 2, :]
    Ec[:, j*n:(j+1)*n] = np.sum(traj[:, 3:, :]**2, axis=1) * m_Rb87 / 2
    Ep[:, j*n:(j+1)*n] = pot(x, y, z)

    filt[j*n:(j+1)*n] = trapped(x[-1], y[-1], z[-1])

Ec = Ec[:, filt]
Ep = Ep[:, filt]

Ec_min = np.min(Ec, axis=0) / h # Hz
Ep_min = (np.min(Ep, axis=0) - P0 * beta_ponder * Emin) / h # Hz

clip_Ec_min = Ec_min[np.nonzero(Ec_min < 0.5e6)] / 1e3
clip_Ep_min = Ep_min[np.nonzero(Ep_min < 0.5e6)] / 1e3


# =============================================================================
# ########## Plot trajectories ##########
# =============================================================================

ntraj = 5

# 3D plot
fig3d = plt.figure(dpi=200)
ax3d = fig3d.add_subplot(projection='3d')
for k in range(ntraj):
    ax3d.plot(traj[:, 2, k]*1e6, traj[:, 0, k]*1e6, traj[:, 1, k]*1e6)
ax3d.set_xlim(-1e6*(zmax := max(z_R))/2, 1e6*zmax/2)
ax3d.set_ylim(-1e6*(wmax := max(w0))/2, 1e6*wmax/2)
ax3d.set_zlim(-1e6*wmax/2, 1e6*wmax/2)
ax3d.set_xlabel("Z (um)")
ax3d.set_ylabel("X (um)")
ax3d.set_zlabel("Y (um)")

# 2D plot
fig2d = plt.figure(figsize=(9, 3), layout='constrained')
gs = fig2d.add_gridspec(nrows=1, ncols=3)
axs = gs.subplots()
# axs[0].set_xlim(-5e5*w0, 5e5*w0); axs[0].set_ylim(-5e5*w0, 5e5*w0)
axs[0].axis('equal')
axs[0].set_title("xy cut")
axs[0].set_xlabel("X (um)"); axs[0].set_ylabel("Y (um)")
# axs[1].set_xlim(-5e5*w0, 5e5*w0); axs[1].set_ylim(-4e5*z_R, 4e5*z_R)
axs[1].axis('equal')
axs[1].set_title("xz cut")
axs[1].set_xlabel('X (um)'); axs[1].set_ylabel('Z (um)')
# axs[2].set_xlim(-5e5*w0, 5e5*w0); axs[2].set_ylim(-4e5*z_R, 4e5*z_R)
axs[2].axis('equal')
axs[2].set_title("yz cut")
axs[2].set_xlabel('Y (um)'); axs[2].set_ylabel('Z (um)')
for i in range(ntraj):
    axs[0].plot(traj[:, 0, i]*1e6, traj[:, 1, i]*1e6)
    axs[1].plot(traj[:, 0, i]*1e6, traj[:, 2, i]*1e6)
    axs[2].plot(traj[:, 1, i]*1e6, traj[:, 2, i]*1e6)

plt.show()


# =============================================================================
# ########## Plot energy distribution ##########
# =============================================================================

fig, ax = plt.subplots(1, 2, figsize=(8, 3), squeeze=False,
                       sharex=True, sharey=True,
                       dpi=200)

## Kinetic energy
ax[0, 0].set_xlim(0, 80)
ax[0, 0].hist(clip_Ec_min, bins=100)
ax[0, 0].set_xlabel(r"Kinetic energy / h (kHz)", fontsize=11, labelpad=3)
ax[0, 0].set_ylabel(r"Count frequency", fontsize=11, labelpad=3)

## potential energy
ax[0, 1].hist(clip_Ep_min, bins=100)
ax[0, 1].set_xlabel(r"Potential energy / h (kHz)", fontsize=11, labelpad=3)


# =============================================================================
# Save data
# =============================================================================
fpath_out = "./data/"

data_out = {
    'shape': bob.dgrid.shape,

    # Trap parameters
    'gauss_V0': np.array(V0_gauss),
    'gauss_depth': np.array(-V0_gauss/k_B*1e3), # mK
    'waist': w0,
    'z_R': z_R,
    'bob_power': np.array(P0), # W
    'bob_Emin': np.array(P0 * beta_ponder * Emin_Hz), # Hz
    'bob_depth_Hz': np.array((thr_Hz - Emin_Hz) * beta_ponder * P0 ),
    'bob_depth_K': np.array((thr_K - Emin_K) * beta_ponder * P0),

    'T': np.array(temperature),

    'times': times*1e6, # convert to us
    'traj': traj[:1001, :, :30] * 1e6, # convert to um
    'min_Ep': Ep_min, # in Hz
    'min_Ec': Ec_min, # in Hz
    'mean_Ec': np.array(np.mean(Ec)) / h, # in Hz
    'mean_Ep': np.array(np.mean(Ep)) / h, # in Hz
    }

# np.savez_compressed(fpath_out + "BoB_sim_traj.npz", **data_out)
fig3d.savefig(fpath_out + "./BoB_trajectories_3d.png")
fig2d.savefig(fpath_out + "./BoB_trajectories_2d.png")
fig.savefig(fpath_out + "./energy_distribution_in_BoBs.png")
