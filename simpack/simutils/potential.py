# -*- coding: utf-8 -*-
"""
Functions for potential and forces computation.

A <Data_Grid> object is implemented. It consists essentially in a container
for data sampled on a regular grid (think of a BoB profile) with additional
attributes for the physical pixel size, the position of the center, ...
The methods are:
    - <get>, to get the init parameters
    - <save>, to save the data

The functions for potential and forces computation are:
    - <build_potential>, build a function V(x, y, z) representing a potential
      (ie a scalar function) from an expression or a sampling in space
    - <build_forces>, build functions (Fx, Fy, Fz)(x, y, z) representing
      the space derivatives of a scalar function from an expression or a
      sampling in space of the latter.

TODO
- gendoc

"""

from inspect import signature
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sym
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator as rgi


default_interp1d_kw = {
    'method' : 'linear',
    'bounds_error' : False,
    'fill_value' : 0,
    }

# =============================================================================
# Data_Grid class
# =============================================================================

class Data_Grid():

    def __init__(self,
                 data: np.ndarray,
                 pxsize: ArrayLike,
                 loc: ArrayLike):
        """
        Instantiate a Data_Grid object given the sampled data, the pixel
        size `pxsize` and the position of the physical origin `loc` in
        array coordinates.

        The conversion from physical positions to array indices is:
        position = (index - loc) * pxsize

        Parameters
        ----------
        data : np.ndarray
            Array representing sampled values of the data on a regular grid
            of mesh pxsize.
        pxsize : ArrayLike (dx, dy, dz)
            The pixel dimensions of the array defining the potential, in m.
        loc : ArrayLike of float (i0, j0, k0)
            The position of the physical origin (position (0, 0, 0)) in the
            array defining the potential, expressed in array coordinates
            (ie pixel units).
            It is not necessarily an integer.

        """
        # Core data info
        self.data = data
        self.pxsize = np.array(pxsize)
        self.loc = np.array(loc)

        # Data array attributes
        self.dtype = data.dtype
        self.shape = np.array(data.shape)

        # Data grid extent
        self.extent = (self.shape - 1) * self.pxsize


    def __getitem__(self, idx):
        return self.data[idx]


    def __repr__(self,):
        data_repr = (f"array({'['*self.data.ndim}...{']'*self.data.ndim}, "
                     f"dtype={self.data.dtype})")
        repr_ = (f"Data_Grid(data={data_repr},"
                 f"\n          pxsize={repr(self.pxsize)},"
                 f"\n          loc={repr(self.loc)})")
        return repr_


    def get_dict(self, *, keyPrefix: str = "")-> dict:
        """
        Get the parameters necessary to init the Data_Grid as a dict.
        """
        return {keyPrefix + 'data': self.data,
                keyPrefix + 'pxsize': self.pxsize,
                keyPrefix + 'loc': self.loc,}


    def save(self, file: str | Path):
        """
        Save the Data_Grid as a numpy compressed archive.
        Only the necessary attributes are saved.

        Parameters
        ----------
        file : str or pathlib.Path
            The filename where the data will be saved.
        """
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        data_out = self.get_dict()
        np.savez_compressed(file, **data_out)
        print(f"Data_Grid saved at: {str(file)}")


    def get_grid_points(self,
                        axes: tuple[int, ...] | None = None
                        )-> tuple[np.ndarray, ...]:
        """
        TODO doc
        Get data grid points
        used as is to compute scipy regular grid interpolator
        """
        if axes is None:
            axes = range(len(self.shape))
        gp = [np.linspace(-self.pxsize[i] * self.loc[i],
                          self.pxsize[i] * (self.shape[i]-1 - self.loc[i]),
                          self.shape[i], endpoint=True)
              for i in axes]
        return tuple(np.round(p, decimals=15) for p in gp)


    def get_section(self, posIdx: tuple[int, ...] = None)-> tuple:
        """
        TODO doc

        Parameters
        ----------
        posIdx : tuple[int], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        tuple extent, (sect_xy, sect_xz, sect_yz)
            extent : np.ndarray
            sections : np.ndarray

        """
        # Default position index
        if posIdx is None:
            posIdx = self.loc.astype(int)
        # Get the sections
        sect_xy = self.data[:, :, posIdx[2]]
        sect_xz = self.data[:, posIdx[1], :]
        sect_yz = self.data[posIdx[0], :, :]
        # Compute the extent
        extent = np.array([-self.loc*self.pxsize,
                           (self.shape-1-self.loc)*self.pxsize]
                          ).transpose()

        return extent, (np.copy(sect_xy), np.copy(sect_xz), np.copy(sect_yz))


# =============================================================================
# Potentials and forces
# =============================================================================

def build_potential(potential: Callable | Data_Grid,
                    offset: tuple[Real, ...] = (0., 0., 0.),
                    amplitude: Real = 1,
                    **interp1d_kw)-> Callable:
    """
    Build a function that gives the potential V(x, y, z) as a function
    of the position (x, y, z).

    Two cases are possible:
    - The potential is given as an array that represents the potential
      sampled on a regular grid.
      In that case, the  array is converted to a linear interpolator
      (scipy RegularGridInterpolator).
    - The potential is given as a function that takes sympy symols as
      arguments and yields a sympy expression.
      In that case, a lambdified function is returned.

    The potential can be scaled by a global amplitude factor and offset.

    Parameters
    ----------
    potential : Data_Grid or Callable
        Object describing the potential. It can be of two kinds:
        - A Data_Grid object representing sampled values of the potential
          on a regular grid of mesh pxsize.
        - A function V(x, y, z) giving the value of the potential
          at position (x, y, z).
          This is for completeness, since in this case one already
          has the potential.
    offset : tuple[Real], optional
        Offset applied to the potential: V(x) = potential(x - offset).
        The default is (0., 0., 0.), no offset.
    amplitude : Real, optional
        Global amplitude factor for the potential: V = amplitude * potential
        The default is 1, no scaling.
    **interp1d_kw : Any
        kwargs passed to scipy RegularGridInterpolator.
        The defaults are:
            - method : 'linear',
            - bounds_error : False,
            - fill_value : 0.

    Raises
    ------
    TypeError
        If the potential is neither an array nor a callable.

    Returns
    -------
    Callable
        Function V: (x, y, z) -> V(x, y, z). It takes two forms:
        - If the potential is an array, the function V is a scipy
          RegularGridInterpolator, that performs linear interpolation
          from the sampled values of the potential.
        - If the potential is a callable, the function V is a lambdified
          sympy expression built from the potential, that yield a
          numerical evaluation of an analytical function.

    """
    if isinstance(potential, Data_Grid):
        grid = potential.get_grid_points()
        for i, g in enumerate(grid):
            g += offset[i]
        data = amplitude * potential.data
        # Build the interpolator
        kw = default_interp1d_kw | interp1d_kw # set interp1d kw
        rgiV = rgi(grid, data, **kw)
        return lambda x, y, z: rgiV((x, y, z))

    elif callable(potential):
        nbarg = len(signature(potential).parameters) # number of arguments
        # build the symbols
        symb = [sym.Symbol(f'x{i}') for i in range(nbarg)]
        symb_offset = [symb[i] - offset[i] for i in range(nbarg)]
        # transform the function into an expression
        expr = amplitude * potential(*symb_offset)
        # lambdify
        return sym.lambdify(symb, expr, cse=True)

    else:
        raise TypeError(f"Invalid type for potential : {type(potential)}")


def build_force(potential: Callable | Data_Grid,
                offset: tuple[Real, ...] = (0., 0., 0.),
                amplitude: Real = 1,
                **interp1d_kw)-> tuple[Callable]:
    """
    For a given potential, build three functions that give the forces
    Fx, Fy, Fz as a function of the position (x, y, z).

    Two cases are possible:
    - The potential is given as an array that represents the potential
      sampled on a regular grid.
      In that case, the forces are computed as the discrete derivative
      dV/dx[i] = (V[i+1] - V[i-1])/2, and the resulting array is
      converted to an interpolator (scipy RegularGridInterpolator).
    - The potential is given as a function that takes sympy symols as
      arguments and yields a sympy expression.
      In that case, the derivative is computed analytically and a
      lambdified function is returned.

    The force can be scaled by a global amplitude factor and offset.

    Parameters
    ----------
    potential : Data_Grid or callable
        Object describing the potential. It can be of two kinds:
        - A Data_Grid object representing sampled values of the potential
          on a regular grid of mesh pxsize.
        - A function V(x, y, z) giving the value of the potential
          at position (x, y, z).
    offset : tuple[Real], optional
        Offset applied to the potential before converting to force.
        F(x) = - deriv[potential](x - offset)
        The default is (0., 0., 0.), no offset.
    amplitude : Real, optional
        Global amplitude factor for the force.
        F = - amplitude * deriv[potential]
        The default is 1, no scaling.
    **interp1d_kw : Any
        kwargs passed to scipy RegularGridInterpolator.
        The defaults are:
            - method : 'linear',
            - bounds_error : False,
            - fill_value : 0.

    Raises
    ------
    TypeError
        If the potential is neither a Data_Grid nor a callable.

    Returns
    -------
    tuple (Fx, Fy, Fz) of functions Fi: (x, y, z) -> Fi(x, y, z)
        - If the potential is an array, the functions Fi are scipy
          RegularGridInterpolator, that perform linear interpolation
          from the sampled values of the forces.
        - If the potential is a callable, the functions Fi are
          lambdified sympy expressions built from the expression of
          the forces, that yields a numerical evaluation of the
          analytical expression.

    """
    if isinstance(potential, Data_Grid):
        xpx, ypx, zpx = potential.pxsize
        pot = amplitude * potential.data

        grid = potential.get_grid_points()
        for i, g in enumerate(grid):
            g += offset[i]

        # Fx, bulk and edge derivatives
        Fx = (np.roll(pot, 1, axis=0) - np.roll(pot, -1, axis=0)) / (2*xpx)
        Fx[0, :, :] = (pot[0, :, :] - pot[1, :, :]) / xpx
        Fx[-1, :, :] = (pot[-2, :, :] - pot[-1, :, :]) / xpx
        # Fy, bulk and edge derivatives
        Fy = (np.roll(pot, 1, axis=1) - np.roll(pot, -1, axis=1)) / (2*ypx)
        Fy[:, 0, :] = (pot[:, 0, :] - pot[:, 1, :]) / ypx
        Fy[:, -1, :] = (pot[:, -2, :] - pot[:, -1, :]) / ypx
        # Fz, bulk and edge derivatives
        Fz = (np.roll(pot, 1, axis=2) - np.roll(pot, -1, axis=2)) / (2*zpx)
        Fz[:, :, 0] = (pot[:, :, 0] - pot[:, :, 1]) / zpx
        Fz[:, :, -1] = (pot[:, :, -2] - pot[:, :, -1]) / zpx

        # Build the interpolators
        kw = default_interp1d_kw | interp1d_kw # set interp1d kw
        rgiFx = rgi(grid, Fx, **kw)
        rgiFy = rgi(grid, Fy, **kw)
        rgiFz = rgi(grid, Fz, **kw)
        return (lambda x, y, z: rgiFx((x, y, z)),
                lambda x, y, z: rgiFy((x, y, z)),
                lambda x, y, z: rgiFz((x, y, z)))

    elif callable(potential):
        nbarg = len(signature(potential).parameters) # number of arguments
        # build the symbols
        symb = [sym.Symbol(f'x{i}') for i in range(nbarg)]
        symb_offset = [symb[i] - offset[i] for i in range(nbarg)]
        # transform the function into an expression
        expr = amplitude * potential(*symb_offset)
        # build derivative
        dexpr = [- sym.diff(expr, s) for s in symb] # Fx = - dV/dx, etc
        # lambdify the derivative
        F = tuple(sym.lambdify(symb, dexpr[i], cse=True)
                  for i in range(nbarg))
        return F

    else:
        raise TypeError(f"Invalid type for potential : {type(potential)}")


