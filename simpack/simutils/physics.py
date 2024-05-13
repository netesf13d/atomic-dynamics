# -*- coding: utf-8 -*-
"""
Functions for the evaluation of the probability density of the electron of a
circular state. Involved in the computation of the ponderomotive energy.

TODO
- gendoc
- doc
"""

from numbers import Real
from typing import Callable

import numpy as np
import mpmath as mp
from mpmath import power, exp
from mpmath import fac as factorial
from mpmath import sqrt
from numpy import sin
from scipy.integrate import dblquad, tplquad

# precision of the computation: 100 digits
mp.prec = 100
if mp.libmp.BACKEND == "python":
    print("Module mpmath currently uses Python integers internally.\n",
          "Install gmpy (conda install gmpy2) for faster execution of the code.\n",
          "Ensure gmpy is used :\n",
          ">>> import mpmath.libmp\n",
          ">>> mpmath.libmp.BACKEND\n",
          "should give 'gmpy' instead of 'python'\n\n")

from ..config.physics import a


# =============================================================================
# Circular state probability density class
# =============================================================================

class Circ_Prob_Density():

    a = a # reduced Bohr radius as a class attribute

    def __init__(self, n: int, norm: float = None):
        """
        TODO doc

        Parameters
        ----------
        n : int
            DESCRIPTION.
        norm : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.n = n # int
        print("Circ_Proba_Density: initializing callables...", end="")
        # analytical expression in spherical coordinates
        self.p_sph = circprob_sph(n)
        # analytical expression in spherical coordinates
        self.p_cart = circprob_cart(n)
        # gaussian approx in spherical coordinates
        self.p_gauss_sph = circprob_gauss_sph(n, rbound=2*n**2*a)
        # gaussian approx in cartesian coordinates
        self.p_gauss_cart = circprob_gauss_cart(n, bounds=2*n**2*a, norm=norm)
        print(" done")


    def eval_prob_sph(self, r: float, theta: float)-> float:
        """
        Evaluate the analytical probability density of the circular state.
        The evaluation cannot be vectorized, non-vectorizable mpmath
        computation is carried internally.

        Parameters
        ----------
        r, theta : float
            Spherical coordinates. The proba density has no phi dependence.

        """
        return self.p_sph(r, theta)


    def eval_prob_cart(self, x: float, y: float, z: float)-> float:
        """
        Evaluate the analytical probability density of the circular state.
        The evaluation cannot be vectorized, non-vectorizable mpmath
        computation is carried internally.

        Parameters
        ----------
        x, y, z : float
            Cartesian coordinates.

        """
        return self.p_cart(x, y, z)


    def eval_prob_gauss_sph(self,
                             r: np.ndarray,
                             theta: np.ndarray)-> np.ndarray:
        """
        Evaluate the gaussian-approximated probability density of the
        circular state.

        Parameters
        ----------
        r, theta : np.ndarray
            Spherical coordinates. The proba density has no phi dependence.

        """
        return self.p_gauss_sph(r, theta)


    def eval_prob_gauss_cart(self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray)-> np.ndarray:
        """
        Evaluate the gaussian-approximated probability density of the
        circular state.

        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates.

        """
        return self.p_gauss_cart(x, y, z)


    def prob_density_array(self,
                           pxsize: tuple,
                           eval_mode: str = "approximated",
                           optimize_shape: bool = True,
                           shape: tuple = None)-> dict:
        """
        Probability density evaluated on a grid of optimized size.

        The shape is computed internally so as to have a small size while
        retaining most of the probability density.
        TODO doc

        If the mode "optimized" is selected, the output shape components are
        odd.

        Parameters
        ----------
        pxsize : tuple
            DESCRIPTION.
        eval_mode : str {"approximated", "analytical"}, optional
            DESCRIPTION. The default is "approximated".
        optimize_shape : bool, optional
            DESCRIPTION. The default is True.
        shape : tuple, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        pxsz = np.array(pxsize)
        if optimize_shape:
            r = self.n**2 * self.a
            dr = self.n**(3/2) * self.a
            # Those parameters were found to be fine for n > 30 based on
            # an empirical study carried on 2022-07-13 at 2am
            sz = np.array([2.25*(r+dr), 2.25*(r+dr), 4.*dr])
            sh = 2 * ((sz/pxsz).astype(int)//2) + 1
        else:
            if shape is None:
                raise ValueError(
                    "parameter `shape` is required for proba density array "
                    "computation in mode 'raw'")
            sh = np.array(shape)
        loc = (sh-1)/2

        bounds = np.zeros((6,), dtype=float)
        bounds[0::2] = -(sh-1)/2*pxsz
        bounds[1::2] = (sh-1)/2*pxsz

        xyz = np.mgrid[bounds[0]:bounds[1]:sh[0]*1j,
                       bounds[2]:bounds[3]:sh[1]*1j,
                       bounds[4]:bounds[5]:sh[2]*1j]

        if eval_mode == "approximated":
            pgrid = self.p_gauss_cart(*xyz)
        elif eval_mode == "analytical":
            xyz = np.moveaxis(xyz, 0, -1)
            pgrid = np.zeros(xyz.shape[:-1], dtype=float)
            for idx, _ in np.ndenumerate(pgrid):
                pgrid[idx] = self.p_cart(*xyz[idx])
        pgrid /= np.sum(pgrid) # Normalize so that the sum is 1

        return {'data': pgrid,
                'pxsize': pxsize,
                'loc': loc}


# =============================================================================
# Circular state probability density functions
# =============================================================================

def circprob_sph(n: int)-> Callable:
    """
    TODO doc
    Probability density for circular state nC: |phi(r, theta)|^2

    Parameters
    ----------
    n : int
        DESCRIPTION.

    Returns
    -------
    callable
        DESCRIPTION.

    """
    na = n*a
    norm = 1./(np.pi*a**3) * power(n * factorial(n), -2) * power(na, -2*(n-1))
    def cd_func(r, theta):
        return norm * power(r*sin(theta), 2*(n-1)) * exp(-2 * r / na)
    return cd_func


def circprob_cart(n: int)-> Callable:
    """
    TODO doc
    Probability density for circular state nC: |phi(r, theta)|^2

    Parameters
    ----------
    n : int
        DESCRIPTION.

    Returns
    -------
    callable
        DESCRIPTION.

    """
    na = n*a
    norm = 1./(np.pi*a**3) * power(n * factorial(n), -2) * power(na, -2*(n-1))

    def cd_func(x, y, z):
        rho2 = power(x, 2) + power(y, 2)
        r = sqrt(power(x, 2) + power(y, 2) + power(z, 2))
        return norm * power(rho2, (n-1)) * exp(-2 * r / na)
    return cd_func


def circprob_gauss_sph(n: int, rbound: float = None)-> Callable:
    """
    TODO doc
    Gaussian approximation for the probability density of circular state nC:
        |phi(r, theta)|^2

    Parameters
    ----------
    n : int
        DESCRIPTION.

    Returns
    -------
    callable
        DESCRIPTION.

    """
    if rbound is None:
        rbound = 2*n**2*a
    mu_r = a*n*(n-1)
    sigma2_r = a**2*n**3
    mu_theta = np.pi/2
    N = 2*np.pi**2 * a**3 * n**3 * ((n-1)**2 + n/2)
    def cd_func(r, theta):
        return 1/N * np.exp(- (r-mu_r)**2 / sigma2_r) \
               * np.exp(- n * (theta-mu_theta)**2 )

    cd = lambda r, theta: \
        cd_func(r, theta) * 2*np.pi * np.sin(theta) * r**2
    norm, _ = dblquad(cd, 0., np.pi, 0., rbound)
    return lambda r, theta: cd_func(r, theta) / norm


def circprob_gauss_cart(
        n: int,
        bounds: float | list | np.ndarray | tuple = None,
        norm: float = None)-> Callable:
    """
    TODO doc
    Gaussian approximation for the probability density of circular state nC:
        |phi(r, theta)|^2

    Parameters
    ----------
    n : int
        DESCRIPTION.
    bounds : float | list | np.ndarray | tuple, optional
        . The default is None.
    norm : float or None
        . The default is None.

    Returns
    -------
    callable
        DESCRIPTION.

    """
    if bounds is None:
        bounds = 2*n**2*a
    if isinstance(bounds, Real):
        bounds = np.full((6,), bounds, dtype=float)
        bounds[0::2] *= -1

    mu_r = a*n*(n-1)
    sigma2_r = a**2*n**3
    sigma2_z = a**2*n**3
    N = 2*np.pi**2 * a**3 * n**3 * ((n-1)**2 + n/2)

    def cd_func(z, y, x):
        return 1/N * \
            np.exp(- (np.sqrt(x**2 + y**2 + z**2)-mu_r)**2 / sigma2_r) \
            * np.exp(- z**2 / sigma2_z )

    if norm is None:
        norm, _ = tplquad(cd_func, *bounds)
    elif norm < 0:
        raise ValueError(
            f"circ_cd_gauss_cart: norm must be positive, got {norm}")

    return lambda x, y, z: cd_func(z, y, x) / norm





