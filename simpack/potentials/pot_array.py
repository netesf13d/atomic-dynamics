# -*- coding: utf-8 -*-
"""
Module implementing some functions to compute trapping potentials, determine
the traping intensity threshold, computing the interior of the trap,
sving, loading and displaying data.

The <Potential_Array> class implements the following methods.
Private utility methods are:
    - <_init_analysis_container>, 
    - <_chk_args>
    - <_chk_needs>,
    - <_build_callable>,
    - <__call__>,
IO methods are:
    - <_load_dict>,
    - <_load_pa>,
    - <save>,
Get and set:
    - <delete_analysis>
    - <get>
    - <get_analysis>
Computations:
    - <compute_potential>,
    - <regular_sampling> 
Analysis routines:
    - <_threshold_analysis>
    - <_interior_analysis>
    - <_curvature_analysis>
Analysis methods:
    - <analyze>
    - <get_thresholds>
    - <get_interior>
    - <get_curvature>
Get and plot data sections:
    - <get_section>,
    - <plot_section>, 

The functions.
Curvature analysis:
    - <curv_poly1d>, 
    - <fit_poly3>, 


TODO
- gen doc
- docs
"""

import psutil
import warnings
from copy import deepcopy
# from inspect import signature
from itertools import combinations_with_replacement as comb_w_repl
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


try: # try to import cupy
    import cupy
except ImportError:  # import scipy fft routines
    cupy = None

# Set `xp` as an alias for cupy if the module is available, numpy otherwise
def get_xp():
    if cupy is not None:
        return cupy
    return np
xp = get_xp()

from ..simutils import (
    Data_Grid,
    get_fill_threshold, get_connected_comp,
    )
from ..config.config import NB_THR_ITERATIONS


# Get the mamximum memory available to the comutation device, in bytes
try:
    MAXMEM = xp.cuda.Device(0).mem_info[1]
except AttributeError:
    MAXMEM = psutil.virtual_memory().total



# =============================================================================
# Utilities
# =============================================================================


# =============================================================================
# Potential_Array class
# =============================================================================

class Potential_Array():
    
    analyses_kw = {"threshold",
                   "interior",
                   "curvature"}
    
    def __init__(self,
                 source: dict | Data_Grid | Callable | str | Path,
                 shape: ArrayLike = None,
                 pxsize: ArrayLike = None,
                 loc: ArrayLike = None):
        """
        Instanciate a new Potential_Array from a given source and additional
        information
        
        The source can be:
            - Data_Grid or Data_Grid-like dict
              The Potential_Array is instantiated from the Data_Grid info.
            - Potential_Array-like dict
              The Potential_Array is instantiated with the available analyses.
            - str or Path pointing to a .npz numpy compressed archive
              The Potential_Array is instantiated from the dict obtained
              by loading the file according to the two cases above.
            - a function f: x, y, z -> f(x, y, z)
              The arguments `shape`, `pxsize`, `loc` are required.
              The Potential array is instantiated by sampling the function
              on the grid defined by the above arguments.

        Parameters
        ----------
        source : Union[dict, Data_Grid, callable, str, Path]
            .
        shape : ArrayLike of int (shx, shy, shz), optional
            Shape of the created Data_Grid. Used only if source is a callable.
            The default is None.
        pxsize : ArrayLike of float (dx, dy, dz), optional
            Pixel size of the created Data_Grid (in meters). Used only if
            source is a callable. The default is None.
        loc : ArrayLike of float (i0, j0, k0)
            The position of the physical origin (position (0, 0, 0)) of the
            Data_Grid, expressed in array coordinates (ie pixel units).
            It is not necessarily an integer.
            The default is None.

        Raises
        ------
        TypeError
            If the source is provided with incorrect type.
        ValueError
            If source is a callable and arguments `shape`, `pxsize`, `loc`
            are not specified.

        """
        ## General parameters
        self.analyses = set() # set[str]
        
        ## Potential data and analyses
        self.dgrid = None # Data_Grid
        self.analysis = self._init_analysis_container() # dict
        
        ## 
        self._callable = None # function or scipy RegularGridInterpolator
        
        
        ### Prepare the profile
        if isinstance(source, dict): # Load from dict
            if 'SAVE_FLAG' in source.keys(): # from a BoB_Array
                self._load_pa(source)
            else: # Data_Grid dict object origin
                self._load_dict(source)
        elif isinstance(source, Data_Grid):
            self.dgrid = source
        elif callable(source):
            self._chk_args(shape=shape, pxsize=pxsize, loc=loc)
            ext = (np.array(shape) - 1) * np.array(pxsize)
            x0, y0, z0 = np.array(loc) * np.array(pxsize)
            x, y, z = np.mgrid[-x0:ext[0]-x0:shape[0]*1j,
                               -y0:ext[1]-y0:shape[1]*1j,
                               -z0:ext[2]-z0:shape[2]*1j]
            self.dgrid = Data_Grid(source(x, y, z), pxsize, loc)
            self._callable = source
        elif isinstance(source, (Path, str)): # Load from file
            with np.load(source, allow_pickle=False) as f:
                if 'SAVE_FLAG' in f.keys(): # saved BoB_Array
                    self._load_pa(f)
                else: # other file origin
                    self._load_dict(f)
        else:
            raise TypeError("source must be of type dict, Data_Grid, "
                            f"callable or str/Path; got {type(source)}")
        # Compute the callable
        self._build_callable()
    
    
    ########## Utilities ##########
    
    def _init_analysis_container(self, an: dict = None)-> dict:
        """
        TODO update doc
        Initialize the container for analyses.
        """
        analysisDict = {
            ## set by analysing threshold
            'thr': np.array([]), # float
            'thr_err': np.array([]), # float
            ## set by analyzing interior
            'interior': np.array([]), # np.ndarray, same shape as parent entry
            'min': np.array([]), # float
            ## set by analyzing the curvature
            'curvature': np.array([]), # np.ndarray, shape (3,)
            'poly1d_coef': np.array([]), # np.ndarray, shape (3, deg+1)
            'poly1D_domain': np.array([]), # np.ndarray, shape (3, 2)
            'poly1d_window': np.array([]), # np.ndarray, shape (3, 2)
            }
        
        if an is None:
            return analysisDict
        elif isinstance(an, dict):
            an.update(analysisDict)
            return
    
    
    def _chk_args(self, **kwargs):
        """
        Check that the arguments are specified (ie they are not None).

        Raises
        ------
        ValueError
            If one of the arguments shape, pxsize, or nb_bob is not
            specified.

        """
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(f"argument `{k}` not specified")

    
    def _chk_needs(self, caller: str,
                   analyses: str | set = None,)-> None:
        """
        TODO doc

        Parameters
        ----------
        needs : str | set
            DESCRIPTION.
        caller : str
            DESCRIPTION.

        Raises
        ------
        AttributeError
            DESCRIPTION.
        """
        # Check analyses
        if analyses is None:
            analyses = set()
        if isinstance(analyses, str):
            analyses = {analyses}
        # Check
        if not analyses.issubset(self.analyses_kw):
            raise ValueError(
                f"analyses {analyses} must be in {self.analyses_kw}")
        if not analyses.issubset(self.analyses):
            raise AttributeError(
                f"{caller}: {analyses.difference(self.analyses)} "
                "entries not available")

    
    def _build_callable(self):
        """
        Build the callable associated to the potential array.
        The function does nothing if the callable is already available
        (for instance if given at init).
        If not already set, the callable is built from the Data_Grid as a
        scipy RegularGridInterpolator.
        
        This is used for potential computation and sub-sampling of the
        BoB profile.
        """
        if self._callable is not None:
            return
        else:            
            _rgi = rgi(self.dgrid.get_grid_points(),
                       self.dgrid.data,
                       method='linear',
                       bounds_error=True)
            self._callable = lambda x, y, z: _rgi((x, y, z))
            return
    
    
    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray)-> np.ndarray:
        """
        Evaluate the potential at positions x, y, z.

        Parameters
        ----------
        x, y, z : np.ndarray
            The coordinates at which to evaluate the fuunction.

        Returns
        -------
        np.ndarray
            The potential evaluated at position x, y, z.

        """
        return self._callable(x, y, z)
    
    
    ########## Loading and saving ##########
    
    def _load_dict(self, source: dict):
        """
        Load Data_Grid from dict.
        """
        self.dgrid = Data_Grid(source['data'],
                               source['pxsize'],
                               source['loc'])
    

    def _load_pa(self, source: dict):
        """
        TODO doc
        Load Potential_Array from dict

        Parameters
        ----------
        source : dict
            DESCRIPTION.

        """
        ##### General parameters
        self.analyses = set(source['analyses'])
        ##### Entries
        ## BoB data
        self.dgrid = Data_Grid(
            data = source['potential.data'],
            pxsize = source['potential.pxsize'],
            loc = source['potential.loc'])
        ##### Analyses
        for key in source.keys():
            if (k:=key.split('.'))[0] == "analysis":
                self.analysis[k[1]] = source[key]
        print("Potential array loaded successfully")
    
    
    def save(self, file: str):
        """
        TODO doc
        Save the Potential_Array.

        Parameters
        ----------
        file : str or pathlib.Path
            The filename where the data will be saved.

        Returns
        -------
        None.

        """
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        ##### General parameters
        data_out = {
            'SAVE_FLAG': np.array(True), # to tell __init__ that the data comes from a BoB_Array
            'analyses': np.array([an for an in self.analyses])}
        ##### Potential array
        data_out.update(self.dgrid.get_dict(keyPrefix="potential."))
        ##### Analyses
        data_out.update({"analysis." + key: np.array(val)
                         for key, val in self.analysis.items()})
        ##### Save
        np.savez_compressed(file, **data_out)
        print(f"Potential array saved at: {str(file)}")
    
    
    ########## Get and set entries/analyses ##########        
        
    def delete_analysis(self, analysis: str):
        """
        Delete the given analysis.
        The analysis is specified as entry_kw.analysis_kw with:
            - entry_kw in {"bob", "potential"}
            - analysis_kw in {"localization", "threshold",
                              "interior", "curvature"}
        """
        # Reset analsis
        if analysis == "all":
            self.analysis = self._init_analysis_container()
            self.analyses = set()
            print("all analyses deleted")
            return
        elif analysis == "threshold":
            self.analysis['thr'] = np.array([])
            self.analysis['thr_err'] = np.array([])
        elif analysis == "interior":
            self.analysis['interior'] = np.array([])
            self.analysis['min'] = np.array([])
        elif analysis == "curvature":
            self.analysis['curvature'] = np.array([])
            self.analysis['poly1d_coef'] = np.array([])
            self.analysis['poly1d_domain'] = np.array([])
            self.analysis['poly1d_window'] = np.array([])
        self.analyses.remove(analysis)
        print(f"`{analysis}` analysis deleted")
    
    
    def get(self, entry: str)-> Data_Grid:
        """
        Get a copy of the Data_Grid object associated to the potential array.
        """
        return deepcopy(self.dgrid)
    
    
    def get_analysis(self, analysis: str):
        """
        Get a copy of the given analysis, if available.
        """
        # Checks
        self._chk_needs(caller="get_analysis", analyses=analysis)
        # get and copy analysis
        if analysis == "threshold":
            return self.analysis['thr'], self.analysis['thr_err']
        elif analysis == "interior":
            return np.copy(self.analysis['interior']), self.analysis['min']
        elif analysis == "curvature":
            return (np.copy(self.analysis['curvature']),
                    np.copy(self.analysis['poly1d_coef']),
                    np.copy(self.analysis['poly1d_domain']),
                    np.copy(self.analysis['poly1d_window']))
    
    
    ########## Computation and sampling ##########
    
    
    def compute_potential(self,
                          pxsize: ArrayLike,
                          probability_density: Data_Grid,
                          axis: int = 0,
                          symmetries: tuple[int] = (0, 1, 2))-> Data_Grid:
        """
        TODO doc

        Parameters
        ----------
        pxsize : tuple[float]
            DESCRIPTION.
        axis : int, optional
            DESCRIPTION. The default is 0.
        symmetries : tuple[int], optional
            DESCRIPTION. The default is (0, 1, 2).

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        potential : TYPE
            DESCRIPTION.

        """
        ## Properly align the proba density array
        newaxes = [axis if i == 2 else 2 if i == axis else i
                   for i in range(3)]
        pd = np.transpose(probability_density.data, axes=newaxes)
        pd_pxsz = probability_density.pxsize[newaxes]
        pd_sh = np.array(pd.shape)
        pd = xp.asarray(pd) # convert to cupy array if possible
        
        ## Enforce an integer ratio of pixel sizes between potential and pd
        ## Otherwise the remaining computation will be a pain in the ass
        pot_pxsz = np.array(pxsize)
        px_ratio = np.round(pot_pxsz / pd_pxsz, decimals=5)
        if not np.all(np.isclose(px_ratio, px_ratio.astype(int))):
            raise ValueError(
                "non integer ratio of pixel sizes for potential / "
                f"probability density: {px_ratio}")
        
        ## Compute the shape of the potential and initialize it
        pot_sh = ((self.dgrid.extent - (pd_sh-1)*pd_pxsz) / pot_pxsz).astype(int)
        pot_sh = 2 * ((pot_sh + 1) // 2) - 1 # adjust to get odd shape
        pot_loc = ((pot_sh[0]-1)/2, (pot_sh[1]-1)/2, (pot_sh[2]-1)/2)
        potential = np.zeros(pot_sh, dtype=float)
        print(f"Building potential with shape {pot_sh}")
        
        ## Grid points pertaining to the probability density
        pdx, pdy, pdz = probability_density.get_grid_points(axes=newaxes)
        ## Steps
        stepi = min(int(px_ratio[0]), pd_sh[0])
        stepj = min(int(px_ratio[1]), pd_sh[1])
        stepk = min(int(px_ratio[2]), pd_sh[2])
        
        ## Manage symmetries to get the potential grid points
        pot_nb = [(pot_sh[i]+1)//2 if i in symmetries else pot_sh[i]
                  for i in range(3)]
        pot_min = [0. if i in symmetries else -pot_loc[i]*pot_pxsz[i]
                   for i in range(3)]
        i0, j0, k0 = [(pot_sh[i]-1)//2 if i in symmetries else 0
                      for i in range(3)]
        potx, poty, potz = [
            np.linspace(pot_min[i], pot_loc[i]*pot_pxsz[i], pot_nb[i])
            for i in range(3)]
        
        ### set the grid points for bob RGI sampling
        xx = potx[:, None] + pdx[None, :]
        xx = np.unique(np.round(xx, decimals=15))
        yy = poty[:, None] + pdy[None, :]
        yy = np.unique(np.round(yy, decimals=15))
        
        ## z-slabs: compute size
        mem_xyslc = 6 * 8 * xx.size * yy.size # overhead * float64_size * xy_size
        rgi_nbz = psutil.virtual_memory().available / mem_xyslc
        pot_nbz_per_slab = int(1 + np.floor((rgi_nbz - pd_sh[-1])/stepk))
        if pot_nbz_per_slab <= 0:
            raise MemoryError("not enough memory for potential computation")
        
        ## z-slabs: compute sampling points
        nbzz = int(np.ceil(pot_nb[-1] / pot_nbz_per_slab))
        k0_z = [k0 + j*pot_nbz_per_slab for j in range(nbzz)]
        potz_z = [potz[j*pot_nbz_per_slab:(j+1)*pot_nbz_per_slab]
                  for j in range(nbzz)]
        
        for kk in range(nbzz):
            zz = potz_z[kk][:, None] + pdz[None, :]
            zz = np.unique(np.round(zz, decimals=15))
            
            ## sample the bob profile and cupyfy if possible
            x = xx.reshape((-1, 1, 1))
            y = yy.reshape((1, -1, 1))
            z = zz.reshape((1, 1, -1))
            slab = xp.asarray(self.__call__(x, y, z))
            
            ## compute the convolution
            for i, j, k in np.ndindex(pot_nb[0], pot_nb[1], len(potz_z[kk])):
                # print(i, j, k)
                block = slab[i*stepi: i*stepi + pd_sh[0],
                             j*stepj: j*stepj + pd_sh[1],
                             k*stepk: k*stepk + pd_sh[2]]
                potential[i0+i, j0+j, k0_z[kk]+k] = xp.sum(pd*block)
        
        ## Fill the remaining values from symmetries
        for ax in symmetries:
            source_slc = [slice(None), slice(None), slice(None)]
            source_slc[ax] = slice((pot_sh[ax]+1)//2, None, 1)
            tgt_slc = [slice(None), slice(None), slice(None)]
            tgt_slc[ax] = slice((pot_sh[ax]-1)//2 - 1, None, -1)
            potential[tuple(tgt_slc)] = potential[tuple(source_slc)]
        
        return Data_Grid(potential, pot_pxsz, pot_loc)
    
    
    def regular_sampling(self,
                         pxsize: ArrayLike,
                         shape: ArrayLike,
                         rloc: ArrayLike)-> Data_Grid:
        """
        TODO doc
        rloc is relative to the loc of the parent entry

        Parameters
        ----------
        entry : str
            DESCRIPTION.
        pxsize, shape, rloc : ArrayLike
            DESCRIPTION.

        Returns
        -------
        Data_Grid
            The (regularly) sampled data.

        """
        out = Data_Grid(np.zeros(shape, dtype=np.float64), pxsize, rloc)
        xx, yy, zz = out.get_grid_points()
        x = xx.reshape((-1, 1, 1))
        y = yy.reshape((1, -1, 1))
        z = zz.reshape((1, 1, -1))
        out.data = self.__call__(x, y, z)
        return out
    
    
    ########## Core analysis routines ##########    
    
    def _threshold_analysis(self):
        """
        TODO doc

        Raises
        ------
        AttributeError
            DESCRIPTION.

        """
        # Compute thresholds        
        node = tuple(self.dgrid.loc.astype(int))
        success, thr, thr_error = get_fill_threshold(
            iprof=self.dgrid.data, node=node, nbit=NB_THR_ITERATIONS)
        if success: # valid threshold found
            self.analysis['thr'] = np.array(thr)
            self.analysis['thr_err'] = np.array(thr_error)
        else: # threshold not found (maybe the profile does not have one)
            warnings.warn("threshold analysis: threshold not found",
                          RuntimeWarning)
            self.analysis['thr'] = np.array(np.nan)
            self.analysis['thr_err'] = np.array(np.nan)
        self.analyses.add("threshold")
    
    
    def _interior_analysis(self):
        """
        TODO doc
        Requires `threshold` analysis.
        """
        # checks
        self._chk_needs(caller="_interior_analysis", analyses={"threshold"})
        # Compute interior
        node = tuple(self.dgrid.loc.astype(int))
        thr = self.analysis['thr']
        # Compute
        if np.isnan(thr): # threshold was not found
            warnings.warn("get_interior: invalid threshold to get_interior",
                          RuntimeWarning)
            self.analysis['min'] = np.array(np.nan)
            self.interior = np.zeros_like(self.dgrid.data, dtype=bool)
        else:
            fprof = self.dgrid.data <= thr
            temp = get_connected_comp(fprof=fprof, node=node, mask=None)
            self.analysis['interior'] = temp
            self.analysis['min'] = np.min(self.dgrid.data[temp])
        self.analyses.add("interior")
    
    
    def _curvature_analysis(self,
                            extent: np.ndarray = [0.9e-6, 0.9e-6, 1.8e-6]):
        """
        TODO doc

        Parameters
        ----------
        entry : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        deg = 6
        ### Get positions
        sh = (np.array(extent) / self.dgrid.pxsize).astype(int) // 2
        
        
        self.analysis['curvature'] = np.full((3,), np.nan, dtype=float)
        self.analysis['poly1d_coef'] = np.full((3, deg+1), np.nan, dtype=float)
        self.analysis['poly1d_domain'] = np.full((3, 2), np.nan, dtype=float)
        self.analysis['poly1d_window'] = np.full((3, 2), np.nan, dtype=float)
        # Compute samples and positions grid along x, y, z
        for j in range(3):
            shp = np.zeros_like(sh, dtype=int)
            shp[j] = sh[j]
            val = self.regular_sampling(
                self.dgrid.pxsize, shape=2*shp+1,
                rloc = shp)
            x = val.get_grid_points()[j]
            y = val.data.flatten()
            success, curv, poly = curv_poly1d(x, y, deg=deg)
            if success: # success
                self.analysis['curvature'][j] = curv
                self.analysis['poly1d_coef'][j, :] = poly.coef
                self.analysis['poly1d_domain'][j, :] = poly.domain
                self.analysis['poly1d_window'][j, :] = poly.window
            else: # failure
                warnings.warn(f"curvature analysis: curvature {j} fit failed",
                              RuntimeWarning)
        # val = self.regular_sampling(
        #     entry, ent.pxsize, shape=2*sh+1, rloc = ent.loc - loc + sh)
        # xx, yy, zz = val.get_grid_points()
        # xx = xx.reshape((-1, 1, 1))
        # yy = yy.reshape((1, -1, 1))
        # zz = zz.reshape((1, 1, -1))
        # pnames, res = fit_poly3d((xx, yy, zz), val.data, deg=3)
        # if res.success: # minimization was a success
        #     idx = [pnames.index(pn) for pn in ["Ax", "Ay", "Az"]]
        #     an['curvature'][i, :] = res.x[idx]
        # else: # minimization failure
        #     print(f"curvature analysis: curvature {i} minimization failed")
        #     an['curvature'][i, :] = np.nan
        self.analyses.add("curvature")
    
    
    ########## Analysis ##########
    
    def analyze(self, analysis: str, **kw):
        """
        Wrapper function to perform the given analysis on the given entry.
        entry : {"bob", "potential"}
        analysis : {"localization", "threshold", "interior", "curvature"}
        An error is raised otherwise.
        """
        if analysis == "threshold":
            self._threshold_analysis()
        elif analysis == "interior":
            self._interior_analysis()
        elif analysis == "curvature":
            self._curvature_analysis(**kw)
        else:
            raise ValueError(f"`{analysis}` analysis not available")
    

    def get_threshold(self)-> tuple:
        """
        Get the threshold analysis results on given entry.
        Threshold analysis is carried automatically is not already done.
        Requires `localization` analysis.
        
        Returns
        -------
        thr : 1D np.ndarray
            thr[i] is the trapping threshold of BoB i obtained by dichotomy.
            Corresponds to the highest value of the intensity such that the
            interior of the entry (< threshold) is not connected to the
            exterior.
        thr_err : 1D np.ndarray
            Error on the threshold. The true threshold is in the interval
            [threshold, threshold + threshold_error]
        """
        try: # thresholds already computed, return the values
            return self.get_analysis("threshold")
        except AttributeError:
            self._threshold_analysis()
            return self.get_analysis("threshold")
    
    
    def get_interior(self)-> tuple:
        """
        Get the interior analysis results on given entry.
        Interior analysis is carried automatically is not already done.
        Requires `localization` and `threshold` analysis.
        
        Returns
        -------
        interior : 3D np.ndarray[bool]
            Interior of the entry. The region where entry.data < thr.
        min : 1D np.ndarray
            Minimum of the entry data in the region defined by the interior.
            min = min(entry.data[intterior])
        """
        try: # interior already computed, return the values
            return self.get_analysis("interior")
        except AttributeError:
            self._interior_analysis()
            return self.get_analysis("interior")
    
    
    def get_curvature(self)-> np.ndarray:
        """
        Get the curvature analysis results on given entry.
        Curvature analysis is carried automatically is not already done.
        Requires `localization` analysis.
        
        Returns
        -------
        curvature : 2D np.ndarray
            curvature[i, :] are the x, y, z components of the curvature of
            i-th BoB.
        """
        try: # interior already computed, return the values
            return self.get_analysis("curvature")
        except AttributeError:
            self._curvature_analysis()
            return self.get_analysis("curvature")
    
    
    ########## Get and plot data ##########
    
    def get_section(self, posIdx: tuple[int] = None,
                    entry: str = "potential")-> tuple:
        """
        TODO doc

        Parameters
        ----------
        posIdx : tuple[int]
            Index of the selected Data_Grid array at which the sections are
            taken.
        entry : str {"potential", "interior"}, optional
            Which entry to get section from.
            - "potential" selects the underlying potential
            - "interior" selects the interior of the trap (determined by
              "interior" analysis)
            The default is "potential".

        Raises
        ------
        ValueError
            - If an incorrect entry is given.
            - If asked for unavailable interior.

        Returns
        -------
        (extent_xy, extent_xz, extent_yz) : tuple[np.ndarray]
            Physical extent of the sections (in meters). For instance,
            extent_xy = [xMin, xMax, yMin, yMax]
        (arrxy, arrxz, arryz) : tuple[np.ndarray]
            The sections of the given entry.
        cutpos : np.ndarray [x0, y0, z0]
            The physical position of the sections.

        """
        # Fetch the data, get the section
        if entry == "potential":
            extent, (arrxy, arrxz, arryz) = self.dgrid.get_section(posIdx)
        elif entry == "interior":
            self._chk_needs(caller="get_section", analyses={"interior"})
            interior = self.analysis['interior']
            dg = Data_Grid(interior, self.dgrid.pxsize, self.dgrid.loc)
            extent, (arrxy, arrxz, arryz) = dg.get_section(posIdx)
        else:
            raise ValueError(
                f"entry must be `potential` or `interior`, not {entry}")
            
        # Compute individual section extents
        extent_xy = extent[[0, 0, 1, 1], [0, 1, 0, 1]]
        extent_xz = extent[[0, 0, 2, 2], [0, 1, 0, 1]]
        extent_yz = extent[[1, 1, 2, 2], [0, 1, 0, 1]]
        
        if posIdx is None:
            posIdx = self.dgrid.loc.astype(int)
        cutpos = (posIdx-self.dgrid.loc) * self.dgrid.pxsize
        
        return ((extent_xy, extent_xz, extent_yz),
                (arrxy, arrxz, arryz),
                cutpos)


    def plot_section(self, posIdx: tuple[int] = None,
                     entry: str = "potential"):
        """
        TODO doc

        Parameters
        ----------
        posIdx : tuple[int]
            DESCRIPTION.
        dataType : str, optional
            DESCRIPTION. The default is "bob".

        Returns
        -------
        fig : matplotlib Figure
            The figure on which the section is plotted.

        """
        ((extent_xy, extent_xz, extent_yz),
         (arrxy, arrxz, arryz),
         cutpos) = self.get_section(posIdx, entry)
        extent_xy *= 1e6; extent_xz *= 1e6; extent_yz *= 1e6
        cutpos *= 1e6
        
        arrxy = np.transpose(arrxy)
        extent_xz = np.roll(extent_xz, 2)
        extent_yz = np.roll(extent_yz, 2)
        if entry == "potential":
            arrxy /= 1e12; arrxz /= 1e12; arryz /= 1e12
            title = "Intensity\n(W/um**2)"
        elif entry == "interior":
            title = ""
        
        ### Build custom colormap
        cmap = "viridis"
        vmin = min(np.min(arrxy), np.min(arrxz), np.min(arryz))
        vmax = max(np.max(arrxy), np.max(arrxz), np.max(arryz))
        ### Plot figure
        fig = plt.figure(figsize=(7, 3.5), dpi=200, clear=True)
        
        ax0 = fig.add_axes(rect=(0.1, 0.2, 0.27, 0.6))
        im = ax0.imshow(arrxy, origin="lower", extent=extent_xy, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax0.text(0.6, 0.95, f"xy cut @ z = {cutpos[2]:.3} um",
                  ha="center", va="center", fontsize=9, c="w",
                  transform=ax0.transAxes)
        ax0.set_xlabel("x (um)")
        ax0.set_ylabel("y (um)")
        
        gs1 = GridSpec(nrows=2, ncols=1, hspace=0.02, figure=fig,
                       left=0.42, bottom=0.18, right=0.85, top=0.84)

        ax1 = gs1.subplots(sharex=True, sharey=True, squeeze=True)
        ax1[0].imshow(arrxz, origin="lower", extent=extent_xz, cmap=cmap,
                      vmin=vmin, vmax=vmax)
        ax1[0].text(0.68, 0.92, f"xz cut @ y = {cutpos[1]:.3} um",
                    ha="center", va="center", fontsize=9, c="w",
                    transform=ax1[0].transAxes)
        ax1[1].imshow(arryz, origin="lower", extent=extent_yz, cmap=cmap,
                      vmin=vmin, vmax=vmax)
        ax1[1].text(0.68, 0.92, f"yz cut @ x = {cutpos[0]:.3} um",
                    ha="center", va="center", fontsize=9, c="w",
                    transform=ax1[1].transAxes)
        ax1[1].set_xlabel("z (um)")
        
        ax2 = fig.add_axes(rect=(0.90, 0.2, 0.02, 0.6))
        cb = fig.colorbar(im, cax=ax2, orientation="vertical",
                          ticklocation="right")
        cb.ax.tick_params(axis="both", labelsize=9)
        ax2.set_title(title, loc="center", y=-0.12, pad=-8)
        
        fig.suptitle(f"{entry} section")
        plt.show()
        return fig                                              





# =============================================================================
# BOB ANALYSIS
# =============================================================================


def curv_poly1d(x: np.ndarray,
               y: np.ndarray,
               deg: int)-> tuple:
    """
    TODO doc

    Parameters
    ----------
    x : 1D np.ndarray
        DESCRIPTION.
    y : 1D np.ndarray
        DESCRIPTION.
    deg : int
        Degree of the fitting polynomial.

    Returns
    -------
    success : bool
        True if the polynomial fit succeeded, False otherwise.
    float
        The curvature at the minimum x0. The coefficiant `A` in
        y = y0 + A(x - x0)^2 + ...

    """
    poly = Polynomial.fit(x, y, deg=deg)
    dpoly = poly.deriv()
    ddpoly = dpoly.deriv()
    if np.any(np.isnan(dpoly.coef)): # fit failed
        return False, np.nan, poly
    else: # fit worked, compute derivative to get local extrema
        droots = dpoly.roots()
    ## Recover the correct minimum from the roots of the derivative
    xmin, xmax = np.min(x), np.max(x)
    filt = np.real([r for r in droots if not np.imag(r)]) # real roots
    filt = filt[np.nonzero((filt > xmin) & (filt < xmax))] # roots in range
    filt = filt[ddpoly(filt) > 0] # minima
    if filt.size > 1:
        warnings.warn(
            "Polynomial fit has more than one minimum in range. "
            "Keeping first minimum.",
            RuntimeWarning)
    try:
        xfit = filt[0]
    except IndexError:
        xfit = np.nan
        warnings.warn("Polynomial fit failed: no minimum in range",
                      RuntimeWarning)
    
    ## Curvature at the minimum
    ddx = ddpoly(xfit) / 2
    
    # xx = np.linspace(xmin, xmax, 101)
    # plt.plot(x, y, marker="o", markersize=3)
    # plt.plot(xx, poly(xx))
    # plt.show()
    
    return True, ddx, poly


def fit_poly3d(pos: tuple[np.ndarray],
               val: np.ndarray,
               deg: int):
    """
    pos elements must be broadcastable to the same shape as val

    Parameters
    ----------
    pos : tuple[np.ndarray]
        DESCRIPTION.
    val : np.ndarray
        DESCRIPTION.
    deg : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not (isinstance(deg, int) and deg > 0):
        raise ValueError(f"`deg` must be a positive integer, got {deg}")
    pos = np.array(pos, dtype=object)
    pnames = ["A" + "".join(j)
               for i in range(1, deg+1)
               for j in comb_w_repl("xyz", i)]
    
    val -= np.min(val)
    relpos = pos**2
    
    def diff_func(par: np.ndarray):
        """
        The cost function.
        The parameters are as follows:
            - x0, y0, z0 = par[0], par[1], par[2]
            - 
        """
        diff = np.copy(val)
        cnt = 0
        for i in range(1, deg+1):
            for j, ind in enumerate(comb_w_repl([0, 1, 2], i)):
                diff -= par[cnt+j] * np.prod(relpos[np.array(ind)])
            cnt += ((i+2)*(i+1))//2
        return diff
    
    def cost_func(par: np.ndarray):
        cost = np.sum(diff_func(par)**2)
        print(cost)
        return cost
    
    # def jacobian(par: np.ndarray):
    #     jac = np.zeros_like(par, dtype=np.float64)
    #     jac[0] = -1
    #     for i in range(1, deg+1):
    #         for j, ind in enumerate(comb_w_repl([0, 1, 2], i)):
    #             cost -= par[cnt+j] * np.prod(relpos[np.array(ind)])
    #     return jac
    
    def hessian(par: np.ndarray):
        hess = np.zeros(2*par.shape, dtype=np.float64)
        hess
        return hess
    
    
    # Initial parameter values
    p0 = np.zeros((len(pnames),), dtype=float)
    cnt = 0
    zmax, vmax = np.max(pos[-1]), np.max(val)
    for i in range(1, deg+1):
        p0[cnt:cnt+((i+2)*(i+1))//2] = 1/10**i * (vmax-p0[0]) / zmax**(2*i)
        cnt += ((i+2)*(i+1))//2
    
    print(p0)
    # print(val.size)
    # bounds = [(np.min(x), np.max(x)) for x in pos] \
    #          + [(0., np.Inf) for i in range(len(pnames) - 3)]
    
    res = minimize(cost_func, p0)
            
    return pnames, res    


def fit_poly3d_dev(pos: tuple[np.ndarray],
               val: np.ndarray,
               deg: int):
    """
    pos elements must be broadcastable to the same shape as val

    Parameters
    ----------
    pos : tuple[np.ndarray]
        DESCRIPTION.
    val : np.ndarray
        DESCRIPTION.
    deg : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not (isinstance(deg, int) and deg > 0):
        raise ValueError(f"`deg` must be a positive integer, got {deg}")
    pos = np.array(pos, dtype=object)
    pnames = ["fmin", "x0", "y0", "z0"]
    for i in range(1, deg+1):
        pnames += ["A" + "".join(j) for j in comb_w_repl("xyz", i)]
    
    val /= np.max(val)
    def cost_func(par: np.ndarray):
        """
        The cost function.
        The parameters are as follows:
            - x0, y0, z0 = par[0], par[1], par[2]
            - 
        """
        cost = np.copy(val) - par[0]
        relpos = (pos-par[1:4])**2
        cnt = 4
        for i in range(1, deg+1):
            for j, ind in enumerate(comb_w_repl([0, 1, 2], i)):
                cost -= par[cnt+j] * np.prod(relpos[np.array(ind)])
            cnt += ((i+2)*(i+1))//2
        print(np.sum(cost**2))
        return np.sum(cost**2)
    
    # def jacobian(par: np.ndarray):
    #     jac = np.zeros_like(par, dtype=np.float64)
    #     jac[0] = -1
    #     for i in range(1, deg+1):
    #         for j, ind in enumerate(comb_w_repl([0, 1, 2], i)):
    #             cost -= par[cnt+j] * np.prod(relpos[np.array(ind)])
    #     return jac
    
    def hessian(par: np.ndarray):
        hess = np.zeros(2*par.shape, dtype=np.float64)
        hess
        return hess
    
    
    # Initial parameter values
    p0 = np.zeros((len(pnames),), dtype=float)
    p0[0] = np.min(val)
    # p0[1:4] = [np.mean(p) for p in pos]
    cnt = 4
    zmax, vmax = np.max(pos[-1]), np.max(val)
    for i in range(1, deg+1):
        p0[cnt:cnt+((i+2)*(i+1))//2] = 0
            # 1/10**i * (vmax-p0[0]) / zmax**(2*i)
        cnt += ((i+2)*(i+1))//2
    
    # print(p0)
    # print(val.size)
    # bounds = [(np.min(x), np.max(x)) for x in pos] \
    #          + [(0., np.Inf) for i in range(len(pnames) - 3)]
    
    res = minimize(cost_func, p0)
            
    return pnames, res
