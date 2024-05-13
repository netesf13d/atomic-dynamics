# -*- coding: utf-8 -*-
"""
Module implementing tests for the various function of the simpack.
Test functions are (their role is obvious):
    - <test_build_potential>

    - <test_rejection_sampling>

    - <test_evolution>

    - <test_circ_proba_density>
    - <test_proba_density_grid>

    - <test_gbeam_sampling>

    - <test_simu_rrtemp>
    - <test_simu_gaussdyn>

TODO
- gendoc
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# simpack in parent directory
pdir = Path(__file__).resolve().parents[1]
if not (simd := str(pdir)) in sys.path:
    sys.path.append(simd)
import simpack as sim


DATA_PATH = "./data/"


# =============================================================================
# Physical parameters
# =============================================================================

from scipy.constants import physical_constants
a0 = physical_constants['atomic unit of length'][0]

## Gaussian beam parameters
beta_dipole = 5.040776837250485e-34 # J/(W/m^2), dipole potential coefficient
#
gauss_trap_wavelength = 821. * 1e-9 # m, trap wavelength
w0 = np.array([1.1 * 1e-6, 1.3 * 1e-6])  # m, waist
LS = 27.1 * 1e6 # Hz, trap depth
# deduced parameters
z_R = np.pi * w0**2 / gauss_trap_wavelength # m
V0_gauss = beta_dipole * LS # J, potential amplitude
gparam = {'V0': V0_gauss, 'waist': w0, 'z_R': z_R, 'dz': 0.}


## BoB parameters
bob_trap_wavelength = 821. * 1e-9 # m, trap wavelength
P0 = 0.02 # W, beam power
offset = (0., 0., 0.) # m, offset with respect to the gaussian beam
# Deduced parameters
beta_ponder = sim.ponderomotive_coef(bob_trap_wavelength) # Ponderomotive strength
V0_bob = beta_ponder * P0 # J, potential amplitude


# =============================================================================
# SIMUTILS - POTENTIAL
# =============================================================================

def test_build_potential(gbeam_params: dict = gparam):
    """
    Test <simutils.potential.build_potential>.

    Plot xy and yz cuts of the following potentials:
    - attractive gaussian potential given by <gbeam_pot_np>
      (direct evaluation of the analytical expression using numpy)
    - attractive gaussian potential from <gbeam_pot_sym>
      (potential built from the sympy expression)
    It should look like two white (attractive) gaussian profiles and a
    black (repulsive) cropped BoB profile.
    """
    print("test <simutils.potential.build_potential>... ", end="")

    # Gauss teweezer potential : partial of numpy function
    pot_np = lambda x, y, z: sim.gbeam_pot_np(x, y, z, **gbeam_params)
    # Gauss tweezer potential from <build_potential>: analytical potential
    pot_sym = sim.build_gbeam_pot(**gbeam_params)

    ## 2D plot
    wmax = max(gbeam_params['waist'])
    x1, y1, z1= np.mgrid[-2*wmax:2*wmax:101*1j,
                         -2*wmax:2*wmax:101*1j,
                          0:0:1j]
    x2, y2, z2 = np.mgrid[0:0:1j,
                          -2*wmax:2*wmax:51*1j,
                          -6*wmax:6*wmax:151*1j]
    fig2d, axs = plt.subplots(2, 2, layout="constrained")
    axs[0, 0].imshow(pot_np(x1, y1, z1), cmap='hot_r')
    axs[0, 0].set_aspect(1)
    axs[0, 0].set_title("xy cut")
    axs[0, 1].imshow(pot_np(x2, y2, z2)[0, :, :], cmap='hot_r')
    axs[0, 1].set_title("yz cut")
    axs[1, 0].imshow(pot_sym(x1, y1, z1), cmap='hot_r')
    axs[1, 0].set_aspect(1)
    axs[1, 0].set_title("xy cut")
    axs[1, 1].imshow(pot_sym(x2, y2, z2)[0, :, :], cmap='hot_r')
    axs[1, 1].set_title("yz cut")
    plt.show()

    print("done\n")


# =============================================================================
# SIMUTILS - SAMPLING
# =============================================================================

def test_rejection_sampling(T: float = 20e-6,
                            nsamples: int = 1000,
                            V0: float = V0_bob):
    """
    Test <simutils.sampling.rejection_sampling>.

    Draw 1000 samples from a BoB potential loaded from a file and plot
    them in scatter plots.
    It should look like a spot, expanding to a cylinder (the BoB's interior)
    at the extremities when T is large.

    Parameters
    ----------
    T : float, optional
        Temperature at which atoms are sampled. The default is 20e-6.

    """
    print("test <simutils.sampling.rejection_sampling>...", end="\n")
    ## Import data
    bob = sim.Potential_Array(source=DATA_PATH + "BoB_pot.npz")
    interior, Emin = bob.get_interior()
    bob = bob.dgrid
    shx, shy, shz = bob.shape
    pxsz = bob.pxsize
    ## Build potential and sample
    bob.data -= Emin
    bobpot = sim.build_potential(bob, amplitude=V0)
    spl = sim.rejection_sampling(nsamples, T, bobpot, interior,
                                 bob.loc, bob.pxsize)

    # 3D plot
    fig3d = plt.figure(dpi=200)
    ax3d = fig3d.add_subplot(projection='3d')
    ax3d.scatter(spl[2, :], spl[0, :], spl[1, :], s=1)
    ax3d.set_xlim(-(shz-1)/2 * pxsz[2], (shz-1)/2 * pxsz[2])
    ax3d.set_ylim(-(shx-1)/2 * pxsz[0], (shx-1)/2 * pxsz[0])
    ax3d.set_zlim(-(shy-1)/2 * pxsz[1], (shy-1)/2 * pxsz[1])
    ax3d.set_xlabel("Z (um)")
    ax3d.set_ylabel("X (um)")
    ax3d.set_zlabel("Y (um)")

    # 2D plot
    fig2d, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].scatter(spl[0, :], spl[1, :], s=1)
    axs[0].axis('equal')
    axs[0].set_title("xy cut")
    axs[0].set_xlabel("X (m)"); axs[0].set_ylabel("Y (m)")
    axs[1].scatter(spl[1, :], spl[2, :], s=1)
    axs[1].axis('equal')
    axs[1].set_xlabel('Y (m)'); axs[1].set_ylabel('Z (m)')
    plt.show()

    print("done\n")


# =============================================================================
# SIMUTILS - EVOLUTION
# =============================================================================

def test_evolution(T: float = 20e-6,
                   gbeam_params: dict = gparam,
                   ntraj: int = 2,
                   nsteps: int = 800,
                   dt: float = 1e-7):
    """
    Test <simutils.evolution.evolution>.

    Draw a few samples from a gaussian beam potential and compute and plot
    their evolution in the trap.
    It should look like cycling trajectories in the xy plane and back and
    forth trajectories in the xz and yz planes.

    Parameters
    ----------
    T : float, optional
        Temperature at which atoms are sampled. The default is 20e-6.

    """
    print("test <simutils.evolution.evolution>... ", end="")

    spl = sim.gbeam_sampling(ntraj, T, **gbeam_params)
    F = sim.build_gbeam_force(**gbeam_params)

    traj = np.empty((ntraj, 3, nsteps+1))
    traj[:, :, 0] = np.copy(1e6 * spl[0:3, :].transpose())
    for i in range(nsteps):
        sim.evolution(spl, dt, F)
        traj[:, :, i+1] = np.copy(1e6 * spl[0:3, :].transpose())

    # 3D plot
    zmax = max(gbeam_params['z_R'])
    wmax = max(gbeam_params['waist'])
    fig3d = plt.figure(dpi=200)
    ax3d = fig3d.add_subplot(projection='3d')
    for k in range(ntraj):
        ax3d.plot(traj[k, 2, :], traj[k, 0, :], traj[k, 1, :])
    ax3d.set_xlim(-1e6*zmax/2, 1e6*zmax/2)
    ax3d.set_ylim(-1e6*wmax/2, 1e6*wmax/2)
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
        axs[0].plot(traj[i, 0, :], traj[i, 1, :])
        axs[1].plot(traj[i, 0, :], traj[i, 2, :])
        axs[2].plot(traj[i, 1, :], traj[i, 2, :])

    plt.show()

    print("done\n")


# =============================================================================
# SIMUTILS - PHYSICS
# =============================================================================

def test_circ_proba_density(n: int = 50):
    """
    Test <simutils.physics.Circ_Prob_Density> prob density evaluation.

    Plot the radial, angular and vertical charge density of circular state n
    and compare with a gaussian approximation.
    """
    print("test <simutils.physics.Circ_Prob_Density> eval... ", end="\n")

    pd = sim.Circ_Prob_Density(n , norm=1.)

    # Compare radial circ charge density with gaussian approx
    r = np.linspace(0, 2*n**2*a0, 201, endpoint=True)
    cd_r = np.zeros_like(r, dtype=float)
    cd2_r = np.zeros_like(r, dtype=float)
    for i, x in enumerate(r):
        cd_r[i] = pd.eval_prob_sph(x, np.pi/2)
        cd2_r[i] = pd.eval_prob_gauss_sph(x, np.pi/2)
    plt.plot(r*1e9, cd_r)
    plt.plot(r*1e9, cd2_r)
    plt.show()

    # Compare angular circ charge density with gaussian approx
    theta = np.linspace(0, np.pi, 201, endpoint=True)
    cd_theta = np.zeros_like(theta, dtype=float)
    cd2_theta = np.zeros_like(theta, dtype=float)
    for i, x in enumerate(theta):
        cd_theta[i] = pd.eval_prob_sph(a0*n**2, x)
        cd2_theta[i] = pd.eval_prob_gauss_sph(a0*n**2, x)
    plt.plot(theta, cd_theta)
    plt.plot(theta, cd2_theta)
    plt.show()

    # Compare vertical circ charge density with gaussian approx
    z = np.linspace(-n**2*a0, n**2*a0, 201, endpoint=True)
    cd_z = np.zeros_like(z, dtype=float)
    cd3_z = np.zeros_like(z, dtype=float)
    r0 = a0*n*(n-1) / np.sqrt(2)
    rz = np.sqrt(2 * r0**2  + z**2)
    thetaz = np.arccos(z/rz)
    for i, x in enumerate(z):
        cd_z[i] = pd.eval_prob_sph(rz[i], thetaz[i])
        cd3_z[i] = pd.eval_prob_gauss_cart(r0, r0, x)
    plt.plot(z*1e9, cd_z)
    plt.plot(z*1e9, cd3_z)
    plt.show()

    print("done\n")


def test_proba_density_grid(n: int = 50,
                             pxsize: tuple = (5e-9, 5e-9, 2e-9)):
    """
    Test <simutils.physics.Circ_Prob_Density> array preparation.

    Test the computation of circ charge density on a grid.
    """
    print("test <simutils.physics.Circ_Prob_Density> grid generation...")

    pd = sim.Circ_Prob_Density(n , norm=1.)
    pd = pd.prob_density_array(pxsize, "analytical", optimize_shape=True)
    pd = sim.Data_Grid(**pd)

    pd_ = sim.Potential_Array(source=pd)
    pd_.plot_section()

    print("done\n")

# =============================================================================
# POTENTIALS - GAUSS
# =============================================================================

def test_gbeam_sampling(T: float = 20e-6,
                        nsamples: int = 1000,
                        gbeam_params: dict = gparam):
    """
    Test <potentials.gauss.gbeam_sampling>.

    Draw 1000 samples from a gaussian beam potential and plot them in
    scatter plots.
    It should look like a cigar along the z direction, expanding at the
    extremities when T is large.

    Parameters
    ----------
    T : float, optional
        Temperature at which atoms are sampled. The default is 20e-6.

    """
    print("test <potentials.gauss.gbeam_sampling>...", end="\n")

    ts0 = time.time_ns()/10e8
    spl = 1e6 * sim.gbeam_sampling(nsamples, T, **gbeam_params)[0:3, :]
    ts1 = time.time_ns()/10e8
    spler = sim.normal_gbeam_sampler(**gbeam_params)
    ts2 = time.time_ns()/10e8
    spl2 = 1e6 * spler(nsamples, T)[0:3, :]
    ts3 = time.time_ns()/10e8
    print(f"{nsamples} rejection samples in {ts1-ts0} s")
    print(f"Normal sampler ready in {ts2-ts1} s")
    print(f"{nsamples} normal samples in {ts3-ts2} s")

    # 3D plot
    zmax = max(gbeam_params['z_R'])
    w0 = gbeam_params['waist']
    fig3d = plt.figure(dpi=200)
    ax3d = fig3d.add_subplot(projection='3d')
    ax3d.scatter(spl[2, :], spl[0, :], spl[1, :], s=1)
    ax3d.set_xlim(-1e6*zmax, 1e6*zmax)
    ax3d.set_ylim(-2e6*w0[0], 2e6*w0[0])
    ax3d.set_zlim(-2e6*w0[1], 2e6*w0[1])
    ax3d.set_xlabel("Z (um)")
    ax3d.set_ylabel("X (um)")
    ax3d.set_zlabel("Y (um)")

    # 2D plot
    fig2d = plt.figure(figsize=(8, 4), layout='constrained')
    gs = fig2d.add_gridspec(nrows=1, ncols=2)
    axs = gs.subplots()
    axs[0].axis('equal')
    axs[0].set_title("xy cut")
    axs[0].set_xlabel("X (um)"); axs[0].set_ylabel("Y (um)")
    axs[0].scatter(spl[0, :], spl[1, :], s=1)
    axs[0].scatter(spl2[0, :], spl2[1, :], s=1)
    axs[1].axis('equal')
    axs[1].set_title("yz cut")
    axs[1].set_xlabel('Y (um)'); axs[1].set_ylabel('Z (um)')
    axs[1].scatter(spl[1, :], spl[2, :], s=1)
    axs[1].scatter(spl2[1, :], spl2[2, :], s=1)
    plt.show()

    print("done\n")


# =============================================================================
# SIMU
# =============================================================================

def test_simu_rrtemp(T: float = 20e-6,
                     nsamples: int = 50000,
                     gbeam_params: dict = gparam,
                     delays: np.ndarray = np.linspace(0.1e-6, 320.1e-6, 41)):
    """
    Test <simu_rrtemp.run_simulation>.

    Simulate a release-recapture experiment.
    """
    print("test <simu_rrtemp.run_simulation>... ", end="")

    gauss_pot = sim.build_gbeam_pot(**gbeam_params)
    samples = sim.gbeam_sampling(nsamples, T, **gbeam_params)

    rr = sim.run_simu_rr((delays,), gauss_pot, samples, 0.)
    plt.plot(delays, rr)
    plt.show()

    print("done\n")


def test_simu_gaussdyn(T: float = 20e-6,
                       nsamples: int = 50000,
                       gbeam_params: dict = gparam,
                       release1: float = 5.2e-6,
                       delays: np.ndarray = np.linspace(0.1e-6, 80.1e-6, 321),
                       release2: float = 22e-6):
    """
    Test <simu_gaussdyn.run_simulation>.

    Simulate release-induced atomic oscillations in a gaussian trap.
    """
    print("test <simu_gaussdyn.run_simulation>...", end="\n")

    F = sim.build_gbeam_force(**gbeam_params)
    V = sim.build_gbeam_pot(**gbeam_params)
    seq = (release1, delays, release2)
    samples = sim.gbeam_sampling(nsamples, T, **gbeam_params)

    ts0 = time.time_ns()/10e8
    osc = sim.run_simu_gaussdyn(seq, V, F, np.copy(samples), 0.)
    ts1 = time.time_ns()/10e8
    print(f"Gauss dynamics simu for {nsamples} samples in {ts1-ts0} s")

    # print(np.argmax(osc[150:180]))
    plt.plot(seq[1], osc)
    plt.ylim(0, 1)
    plt.show()

    ts0 = time.time_ns()/10e8
    times, traj = sim.traj_gaussdyn(
        (20e-6, release1, 38.85e-6, release2, 20e-6), F, samples[:, :30])
    ts1 = time.time_ns()/10e8
    print(f"Gauss dynamics trajectories for {30} samples in {ts1-ts0} s")
    for i in range(30):
        plt.plot(times, traj[:, 1, i])
    plt.show()
    print("done\n")


# =============================================================================
# SCRIPT
# =============================================================================

if __name__ == '__main__':
    test_build_potential()

    test_rejection_sampling(T=20e-6)

    test_evolution(T=20e-6)

    test_circ_proba_density(n=50)
    test_proba_density_grid(n=50)

    test_gbeam_sampling(T=120e-6)

    test_simu_rrtemp()
    test_simu_gaussdyn()
