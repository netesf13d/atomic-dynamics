# -*- coding: utf-8 -*-
"""
An example to show what is possible with the simpack.

This script simulates a parametric excitation sequence, which is a common way
to measure a trap frequency.
An atom is initially trapped in a gaussian optical tweezer. The amplitude of
the potential (ie, the light intensity) is modulated at a given frequency.
When the modulation frequency corresponds to one of the characteristic
frequencies of the trap, the atomic motion amplitude (hence the energy)
increases.

One can note that the excitation frequencies are shifted towards low
frequencies as compared to the value computed from the curveture at the minimum
of the trap. This is the result of the trap anharmonicity.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


# =============================================================================
# Configuration and physical parameters
# =============================================================================


########## physical parameters ##########
# gaussian beam wavelength
lambda_trap = 0.821 * 1e-6 # m
#
wx, wy = 1.1 * 1e-6, 1.25 * 1e-6  # m
# trap depth
LS = 28 * 1e6 # Hz

# derived values
zRx, zRy = np.pi * wx**2 / lambda_trap, np.pi * wy**2 / lambda_trap # m
beta = - LScoef_5S / LScoef_5S5P * h # J/(W/m^2)
V0 = beta * LS # potential amplitude
gparam = {'V0': V0, 'waist': [wx, wy], 'z_R': [zRx, zRy], 'z0': 0.}

omega_x = 2/wx * np.sqrt(V0/m_Rb87)
omega_y = 2/wy * np.sqrt(V0/m_Rb87)
omega_z = np.sqrt(V0/m_Rb87 * (1/zRx**2 + 1/zRy**2))


########## Simulation parameters ##########
## The trajectories of `n` atoms in a gaussian trapping potential are computed
## over a time `t_tot`, with a simulation timestep `dt`.
## The trap depth is modulated by 15% at frequencies `exc_freqs`.

# atomic temperatures
temperature = 14e-6 # K
# samples per repetition
n = 5000
# excitation frequencies
exc_freqs = np.linspace(0, 200e3, 101, endpoint=True) # Hz
# total simulation time
t_tot = 100e-6
# trajectory timestep
dt = 1e-7


# =============================================================================
# Parametric oscillation is a gaussian trap
# =============================================================================

########## Simulation ##########
# initial atomic samples
samples = sim.gbeam_sampling(n, temperature, **gparam)
# unmodulated trapping force
force = (sim.build_gbeam_force(**gparam),)
# unmodulated trapping potential
potential = sim.build_gbeam_pot(**gparam)
# spatial displacement of the trap
r0 = ((0, 0, 0),) # no displacement

# results
Etot_0 = np.mean(m_Rb87/2 * np.sum(samples[3:, :]**2, axis=0)
                 + potential(*samples[:3, :]) - potential(0, 0, 0))
Ec_0 = np.mean(m_Rb87/2 * samples[3, :]**2)
Ec = np.zeros((3, len(exc_freqs)), dtype=float) # kinetic energy
Etot = np.zeros((len(exc_freqs),), dtype=float) # mechanical energy

## run simulation
for i, freq in enumerate(exc_freqs):
    # trap depth oscillation
    A = (lambda t: 1 + 0.15*np.sin(2*np.pi*freq*t),)
    # simulation
    _, traj = sim.sim_traj(samples, force, A, r0, dt=dt, t_tot=t_tot)
    # compute energy
    Ec[:, i] = m_Rb87/2 * np.mean(np.max(traj[:, 3:, :], axis=0), axis=-1)**2
    Etot[i] = np.mean(m_Rb87/2 * np.sum(traj[-1, 3:, :]**2, axis=0)
                      + potential(*traj[-1, :3, :]) - potential(0, 0, 0))
    # print(i)


# =============================================================================
# ########## Plot ##########
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(exc_freqs/1e3, Etot/Etot_0, color='k', label="total energy")

ax.plot(exc_freqs/1e3, Ec[0]/Ec_0, color="tab:blue", label="Ec along x")
ax.axvline(omega_x/np.pi*1e-3, color="tab:blue")

ax.plot(exc_freqs/1e3, Ec[1]/Ec_0, color="tab:orange", label="Ec along y")
ax.axvline(omega_y/np.pi*1e-3, color="tab:orange")

ax.plot(exc_freqs/1e3, Ec[2]/Ec_0, color="tab:green", label="Ec along z")
ax.axvline(omega_z/np.pi*1e-3, color="tab:green")

ax.legend()
ax.set_xlabel(r"Modulation frequency (kHz)", fontsize=11, labelpad=2)
ax.set_ylabel(r"E final / E initial", fontsize=11, labelpad=4)
fig.text(0.08, 0.9, f"wx = {wx*1e6:.2f} um, wy = {wy*1e6:.2f} um")


fpath_out = "./data/"
fig.savefig(fpath_out + "GaussTrap_parametric_excitation.png")

