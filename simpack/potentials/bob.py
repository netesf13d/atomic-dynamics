# -*- coding: utf-8 -*-
"""
Functions to compute BoB profiles, determine the traping intensity threshold,
computing the interior of the trap, saving, loading and displaying data.

The <BoB_Array> class implements the following methods.
IO methods are:
    - <save>, save the data
Computations:
    - <_compute_bobprof>, compute the bob profile 
Get and plot data sections:
    - <get_ifield>,
    - <get_section>,
    - <plot_section>, 


The functions.
BoB profile computation:
    - <ifield>, compute the field on the SLM plane
    - <bob_profile_from_ifield>, compute the BoB profile from the field
      on the SLM plane
    - <bob_profile>, wrapper around the two functions above
Utility functions
    - <_fetch_bob_params>, get default BoB parameters


TODO
- gen doc
- docs
"""

import psutil

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


try: # try to import cupy
    import cupy
    mempool = cupy.get_default_memory_pool()
    from cupyx.scipy.fft import fftshift, fft2
except ImportError:  # import scipy fft routines
    cupy = None
    mempool = None
    from scipy.fft import fftshift
    from scipy.fft import fft2 as scipy_fft2
    workers = psutil.cpu_count(logical=False)
    def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False):
        """Set arg "worker" when using scipy (no such arg in cupy fft2)"""
        return scipy_fft2(x, s, axes, norm, overwrite_x, workers=workers)

# Set `xp` as an alias for cupy if the module is available, numpy otherwise
def get_xp():
    if cupy is not None:
        return cupy
    return np
xp = get_xp()

from ..simutils import Data_Grid
from ..config.physics import wavelength, focal_length, SLM_defaults


# Get the mamximum memory available to the comutation device, in bytes
try:
    MAXMEM = xp.cuda.Device(0).mem_info[1]
except AttributeError:
    MAXMEM = psutil.virtual_memory().total


# =============================================================================
# Utilities
# =============================================================================

def _fetch_bob_params(**kwargs):
    """
    TODO doc, update

    Parameters
    ----------
    **kwargs : TYPE
        lifetime : float, optional

    Returns
    -------
    dict
        DESCRIPTION.

    """
    # fetch laser wavelength
    try:
        wl = kwargs['wavelength']
    except KeyError:
        wl = wavelength
        print(f"using wavelength default value: {wl*1e9:.4} nm")
    # fetch tweezer lens focal length
    try:
        focal = kwargs['focal_length']
    except KeyError:
        focal = focal_length
        print(f"using focal length default value: {focal*1e3:.3} mm")
    # fetch effective waist of beam intensity profile on SLM 
    try:
        waist = kwargs['I_waist']
    except KeyError:
        waist = SLM_defaults['I_waist']
        print(f"using beam waist default value: {waist*1e3:.3} mm")
    # fetch effective offset of beam intensity profile on SLM
    try:
        offset = kwargs['I_offset']
    except KeyError:
        offset = SLM_defaults['I_offset']
        print(f"using beam intensity offset default value: {offset:.3}")
    # fetch SLM pupil radius
    try:
        r_pupil = kwargs['pupil_radius']
    except KeyError:
        r_pupil = SLM_defaults['pupil_radius']
        print(f"using pupil radius default value: {r_pupil*1e3:.3} mm")
    # fetch SLM bob phaseshift radius
    try:
        r_bob = kwargs['r_bob']
    except KeyError:
        r_bob = SLM_defaults['r_bob']
        print(f"using BoB phaseshift radius default value: {r_bob*1e3:.3} mm")
    
    return {
        'wavelength': wl,
        'focal_length': focal,
        'I_waist': waist,
        'I_offset': offset,
        'pupil_radius': r_pupil,
        'r_bob': r_bob
        }


# =============================================================================
# BoB array class
# =============================================================================

class BoB_Array():
    
    def __init__(self,
                 shape: ArrayLike,
                 pxsize: ArrayLike,
                 bobParameters: dict = None,
                 dtype=np.float64,
                 xfrac_exp: int = 4):
        """
        

        Parameters
        ----------
        shape : ArrayLike of int (shx, shy, shz)
            Shape of the resulting BoB profile. The transverse shapes
            shx, shy must be a power of 2.
        pxsize : ArrayLike(dx, dy, dz)
            The physical pixel dimensions of the BoB profile, in meters.
        bobParameters : dict, optional
            Parameters for the BoB profile:
                - `wavelength`
                - `focal_length`
                - `I_waist`
                - `I_offset`
                - `pupil_radius`
                - `r_bob`
            If some parameters are not provided, default values are fetched
            with <fetch_params>.
            The default is None, in which case default parameters are used.
        dtype : TYPE, optional
            Data type for the BoB profile array. The default is FLOAT_TYPE.
        xfrac_exp : int, optional
            DESCRIPTION. The default is XFRAC.

        Returns
        -------
        None.

        """
        ## General parameters
        self.dtype = dtype # float data type for the various entries, eg float32 or float64...
        self.xfrac_exp = xfrac_exp # int
        self.bobParameters = {} # dict
        
        ## BoB data
        self.bob = None # Data_Grid
        
        ### Prepare the profile
        if bobParameters is None:
            self.bobParameters = _fetch_bob_params()
        else:
            self.bobParameters = _fetch_bob_params(**bobParameters)
        self._compute_bobprof(shape, pxsize, xfrac_exp)
    
    
    ########## File IO ##########
    
    def save(self, file: str):
        """
        TODO doc
        Save the BoB array.

        Parameters
        ----------
        file : str or pathlib.Path
            The filename where the data will be saved.

        Returns
        -------
        None.

        """
        self.bob.save(file)

    
    ########## Computation and sampling ##########
    
    def _compute_bobprof(self,
                         shape: ArrayLike,
                         pxsize: ArrayLike,
                         xfrac_exp: int):
        """
        TODO doc

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        # Check that the shape is valid for bob computation
        # The xy shape must be a power of 2 !
        shx, shy, shz = shape
        shx_exp, shy_exp = round(np.log2(shx)), round(np.log2(shy))
        if 2**shx_exp != shx or 2**shy_exp != shy:
            raise ValueError(
                "bob computation: xy shape must be a power of 2, got "
                f"{shx, shy}")

        # Compute bob profile and set localization of the center
        self.bob = Data_Grid(
            pxsize = pxsize,
            loc = np.array([shx/2, shy/2, (shz-1)/2]),
            data = bob_profile(
                pxsize,
                (shx_exp+xfrac_exp, shy_exp+xfrac_exp),
                shz,
                xfrac_exp=(xfrac_exp, xfrac_exp),
                dtype=self.dtype,
                **self.bobParameters)
            )
    
    
    ########## Get and plot data ##########
    
    def get_ifield(self,)-> tuple:
        """
        Get the electric field profile incident on the SLM (see <ifield>
        documentation).
        
        The field components are cropped to the area of the SLM cell.
        For instance, for an SLM cell of area szx x szy and pixel size
        px x py, the resulting shape of the field components is
        shx, shy = szx/px, szy/py.
        shx and shy are enforced as even integers.
        
        Returns
        -------
        (slm_px, slm_py) : tuple[float]
            The pixel size of the field profile components
        amp, phi, trans_factor : 2D np.ndarrays
            The field norm, phase, and translation factor (to be multiplied
            by z to get the profile out of the focal plane)
        
        """
        shx, shy, _ = self.bob.shape
        shx_exp, shy_exp = round(np.log2(shx)), round(np.log2(shy))
        (slm_px, slm_py), amp, phi, trans_factor = \
            ifield(self.bob.pxsize,
                   (shx_exp+self.xfrac_exp, shy_exp+self.xfrac_exp),
                   **self.bobParameters)
        try: # if cupy was used
            amp, phi, trans_factor = amp.get(), phi.get(), trans_factor.get()
        except AttributeError: # numpy was used
            pass
        
        # find shape for extraction
        szx, szy = SLM_defaults['area']
        xshx, xshy = 2*(int(szx/slm_px)//2), 2*(int(szy/slm_py)//2)
        sel = (slice((shx-xshx)//2, (shx+xshx)//2, 1),
               slice((shy-xshy)//2, (shy+xshy)//2, 1),)
        return (slm_px, slm_py), amp[sel], phi[sel], trans_factor[sel]
    
    
    def get_section(self, posIdx: tuple[int] = None):
        """
        TODO doc

        Parameters
        ----------
        posIdx : tuple[int]
            DESCRIPTION.
        dataType : str, optional
            DESCRIPTION. The default is "bob".

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        extent, (arrxy, arrxz, arryz) = self.bob.get_section(posIdx)
            
        # Compute individual section extents
        extent_xy = extent[[0, 0, 1, 1], [0, 1, 0, 1]]
        extent_xz = extent[[0, 0, 2, 2], [0, 1, 0, 1]]
        extent_yz = extent[[1, 1, 2, 2], [0, 1, 0, 1]]
        
        if posIdx is None:
            posIdx = self.bob.loc.astype(int)
        cutpos = (posIdx-self.bob.loc) * self.bob.pxsize
        
        return ((extent_xy, extent_xz, extent_yz),
                (arrxy, arrxz, arryz),
                cutpos)


    def plot_section(self, posIdx: tuple[int] = None):
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
        None.

        """
        ((extent_xy, extent_xz, extent_yz),
         (arrxy, arrxz, arryz),
         cutpos) = self.get_section(posIdx)
        extent_xy *= 1e6; extent_xz *= 1e6; extent_yz *= 1e6
        cutpos *= 1e6
        
        arrxy = np.transpose(arrxy)
        arrxy /= 1e12; arrxz /= 1e12; arryz /= 1e12
        extent_xz = np.roll(extent_xz, 2)
        extent_yz = np.roll(extent_yz, 2)
        title = "Intensity\n(W/um**2)"
        
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
        
        fig.suptitle("bob section")
        plt.show()



# =============================================================================
# BoB profile computation
# =============================================================================

def ifield(pxsize: tuple[float],
           shr_exp: tuple[int],
           dtype=np.float64,
           **kwargs)-> tuple:
    """
    TODO doc

    Parameters
    ----------
    pxsize : tuple[float] (px, py, pz)
        The pixel size in meters.
    shr_exp : tuple[int] (nx, ny), radial shape exponent
        Exponent of the number of pixels for the computation.
        The phase pattern is imprinted on an array of shape (2**nx, 2**ny).
    **kwargs : 
        Parameters for the BoB profile:
            - `wavelength`
            - `focal_length`
            - `I_waist`
            - `I_offset`
            - `pupil_radius`
            - `r_bob`
        If some parameters are not provided, default values are fetched
        with <fetch_params>.

    Returns
    -------
    tuple (slm_pxsize, amplitude, phase, translation_factor)
        slm_pxsize : tuple (slm_px, slm_py)
        amplitude : 2D ndarray
        phase : 2D ndarray
        translation_factor: 2D ndarray

    """
    # Unpack parameters
    px, py, pz = pxsize
    rshx, rshy = 2**shr_exp[0], 2**shr_exp[1]
    
    ########## Physical parameters initialization ##########
    params = _fetch_bob_params(**kwargs)
    wl = params['wavelength']
    focal = params['focal_length']
    waist = params['I_waist']
    offset = params['I_offset']
    r_pupil = params['pupil_radius']
    r_bob = params['r_bob']
    
    ########## Components of the SLM field ##########
    SLMpx, SLMpy = wl*focal/px/rshx, wl*focal/py/rshy

    uu = xp.linspace(
        -SLMpx*(rshx-1)/2, SLMpx*(rshx-1)/2, rshx,
        endpoint=True, dtype=dtype
        )[:, np.newaxis]
    vv = xp.linspace(
        -SLMpy*(rshy-1)/2, SLMpy*(rshy-1)/2, rshy,
        endpoint=True, dtype=dtype
        )[np.newaxis, :]
    # Pupil
    pupil = (((uu**2 + vv**2) / r_pupil**2) < 1).astype(dtype)
    # Mask phase of the circular pi-shift
    phi = np.pi * (((uu**2 + vv**2) / r_bob**2) < 1).astype(dtype) # Pi*heaviside
    # Intensity distribution of the input beam, in the SLM plane
    sqrtI = xp.sqrt(xp.exp(-2 * (uu**2 + vv**2) / waist**2) + offset)
    # Module of the amplitude
    amp = pupil * sqrtI
    # Translation factor, to be multiplied by z shift
    trans_factor = pz * np.pi/wavelength * ((uu**2 + vv**2) / focal**2)
    
    return (SLMpx, SLMpy), amp, phi, trans_factor


def bob_profile_from_ifield(pxsize: tuple[float],
                            shz: int,
                            xfrac_exp: tuple[int],
                            amplitude,
                            phase,
                            translation_factor)-> np.ndarray:
    """
    TODO doc
    Compute a `theoretical` BoB profile.
    
    The computation is carried block by block to avoid memory 
    overload. If possible, the module cupy is used, numpy otherwise.
    `xp` refers to cupy or numpy depending owhich is used.

    Parameters
    ----------
    pxsize : tuple[float] (px, py, pz)
        The pixel size in meters.
    shz : int, z shape
        The number of z-slices of the profile.
    xfrac_exp : tuple[int] (kx, ky)
        Exponent of the fraction along x and y of the total profile that is
        actually kept. Starting from a raw profile slice of shape
        (rshx, rshy), the kept fraction has shape (rshx//2**kx, rshy//2**ky).

    Raises
    ------
    MemoryError
        If there is not enough memory for profile computation.

    Returns
    -------
    S : 3D np.ndarray of shape (2**(nx-kx), 2**(ny-ky), 2**nz)
        The BoB intensity profile, normalized to unit power.

    """
    if mempool is not None:
        mempool.free_all_blocks()

    # Unpack parameters
    dtype = amplitude.dtype
    px, py, pz = pxsize
    amp = amplitude[..., np.newaxis]
    phi = phase[..., np.newaxis]
    trans_factor = translation_factor[..., np.newaxis]
    rshx, rshy, _ = amp.shape
    shx, shy = rshx // 2**xfrac_exp[0], rshy // 2**xfrac_exp[1]
    # slices to extract data from the large FFt computation
    xslice = slice((rshx-shx)//2, (rshx+shx)//2, 1)
    yslice = slice((rshy-shy)//2, (rshy+shy)//2, 1)
    # Initialize BoB profile container
    S = np.empty((shx, shy, shz), dtype=dtype)
    
    ########## Block computation of the FFT ##########
    # Memory required to compute the BoB profile on a plane
    mem_xyslice = 12 * 2*np.dtype(dtype).itemsize * rshx * rshy # overhead * complex_size * shape
    if MAXMEM / mem_xyslice < 1: # Cannot even compute on a plane
        raise MemoryError(
            "not enough memory to compute the profile: "
            f"needs {mem_xyslice}, got {MAXMEM}")
    
    # Set the z slabs for block computation of the profile 
    nz = int(MAXMEM / mem_xyslice)
    zz = []
    for i in range(int(np.ceil(shz/nz))):
        z1 = -(shz-1)/2 + i*nz
        z2 = min(-(shz-1)/2 + (i+1)*nz, (shz+1)/2)
        zz.append(xp.arange(z1, z2, 1., dtype=dtype))
    
    # block computation
    for i, z in enumerate(zz):
        zslice = slice(i*nz, i*nz + len(z), 1) # dz elements except last one
        ##### z-translation and SLM field amplitude #####
        F = amp * xp.exp(1j * (phi + z*trans_factor))
        # Compute fft to get the BoB amplitude
        F = fftshift(F, axes=(0, 1))
        F = fft2(F, axes=(0, 1), overwrite_x=True) #, workers=workers)
        F = fftshift(F, axes=(0, 1))
        F = xp.abs(F)**2
        F = F / xp.sum(F, axis=(0,1)) / (px * py)
        
        # Set the intensity of the slice
        try: # if cupy was used
            S[:, :, zslice] = F[xslice, yslice, :].get()
        except AttributeError: # numpy was used
            S[:, :, zslice] = F[xslice, yslice, :]
        del F
    return S


def bob_profile(pxsize: tuple[float],
                shr_exp: tuple[int],
                shz: int,
                xfrac_exp: tuple[int],
                dtype=np.float64,
                **kwargs)-> np.ndarray:
    """
    Wrapper around functions <ifield> and <bob_profile_from_ifield> to
    compute a bob profile.
    See the respective documentations.
    """
    # Compute ifield 
    _, amp, phi, trans_factor = ifield(
        pxsize, shr_exp, dtype, **kwargs)
    
    # Compute bob profile and set localization of the center
    return bob_profile_from_ifield(
        pxsize, shz, xfrac_exp, amp, phi, trans_factor)

