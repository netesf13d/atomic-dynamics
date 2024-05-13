# -*- coding: utf-8 -*-
"""
Functions to compute the `interior` of a non-trivial trapping potential and the
trap depth.

Functions for filling method are:
    - <_propagate>, 
    - <get_fill_threshold>, 
    - <get_connected_comp>,
Function for escape path method are:
    - <get_escape_path_threshold>

TODO
- gendoc
- 
"""

import warnings
from itertools import cycle

import numpy as np


# =============================================================================
# Filling method
# =============================================================================

def _propagate(target: np.ndarray,
               fprof: np.ndarray,
               nodes: tuple,
               axis: int,):
    """
    From a set of nodes, compute the nodes that are connected to each
    initial node by propagating along the given axis.

    Parameters
    ----------
    target : np.ndarray of bool
        The connected component built up to this point. It is modified
        in-place by the function.
    fprof : np.ndarray of bool
        The filled profile. It is of the form iprof < threshold
        for a given threshold and intensity profile iprof.
    mask : np.ndarray of bool
        Data mask, fprof values are considered only where mask is True.    
    nodes : tuple
        The nodes from which the propagation is done.
    axis : int
        Direction in which the nodes are propagated.

    Returns
    -------
    edgeflag : bool
        True if the propagation of a node went to an edge of the array.
    outflag : bool
        True if all the propagation nodes are outside the filled profile
        (ie fprof[node] is False)
    newnodes : tuple
        A new set of non-redudant nodes to continue the construction
        of the connected component. The format is:
            newnodes[i] an 1D array for i = 0..dim
            newnodes[i][j] = i-th component of the j-th node

    """        
    n = fprof.ndim
    # flags
    edgeflag = False # the edges of the array have been attained
    outflag = True # the nodes are outside the filled component fprof
    
    for node in zip(*nodes):
        if not fprof[node]: # the node is not in the interior
            pass
        else:
            outflag = False
            # propagate in the positive direction
            arrslc1 = tuple([slice(node[i], None, 1) if i == axis
                             else node[i] for i in range(n)])
            imax = np.argmin((fprof)[arrslc1])
            if imax == 0: # propagation went to the edge
                stop = None
                edgeflag = True
            else:
                stop = node[axis] + imax

            # propagate in the negative direction
            arrslc2 = tuple([slice(node[i], None, -1) if i == axis
                             else node[i] for i in range(n)])
            imin = np.argmin((fprof)[arrslc2])
            if imin == 0: # propagation went to the edge
                start = 0
                edgeflag = True
            else:
                start = node[axis] - imin + 1
            
            # new line to fill
            nline = tuple([slice(start, stop, 1) if i == axis
                           else node[i] for i in range(n)])
            target[nline] = True
    
    ## Test whether a potential node (such that target[node] is True)
    ## should be added to the list of new nodes.
    ## A point (i1, i2, ..., in) is accepted as a new node if 
    ## any of the adjacent points (2*n-way) is not yet in the connected
    ## component (ie target is False) and belongs to the filled profile
    ## (fprof is True)
    rd, ru = [], []
    for i in range(n):
        slc = [slice(None)]*n
        slc[i] = [0]
        rd.append(np.roll(np.logical_not(target)*fprof, 1, axis=i))
        rd[i][tuple(slc)] = False
        slc[i] = [-1]
        ru.append(np.roll(np.logical_not(target)*fprof, -1, axis=i))
        ru[i][tuple(slc)] = False
    
    newnodes = np.nonzero(target)
    nodes_mask = np.nonzero((sum(ru) + sum(rd))[newnodes])
    newnodes = tuple(nn[nodes_mask] for nn in newnodes)
    return edgeflag, outflag, newnodes


def get_fill_threshold(iprof: np.ndarray,
                       node: tuple[int],
                       nbit: int = 12)-> tuple[float]:
    """
    Compute the intensity threshold defining the interior of the trap
    intensity profile iprof, by filling it until it flows outside.
    
    This threshold is defined such that it is not possible to escape
    from the center of the trap without reaching a point of intensity
    > threshold.
    
    The function converges toward this threshold by dichotomy.
    
    The case in which the profile actually has no trapping region
    (ie its laplacian is negative) is managed by the flag outflag

    Parameters
    ----------
    iprof : np.ndarray
        Intensity profile of the trap.
    node : tuple[int]
        Array index in the trap interior.
    nbit : int
        Number of dichotomy iterations.

    Returns
    -------
    threshold : float
        Threshold obtained by dichotomy. Corresponds to the highest
        value of the intensity such that the interior of the trap
        (< threshold) is not connected to the exterior.
    threshold_error : float
        Error on the threshold. The true threshold is in the interval
        [threshold, threshold + threshold_error]

    """
    # Initialize dichotomy loop
    dicho_high = [np.max(iprof)] # open trap thresholds
    dicho_low = [np.min(iprof)] # closed trap thresholds
    success = False
    
    for i in range(nbit):
        # initialization of the threshold
        thr = (dicho_high[-1] + dicho_low[-1]) / 2
        fprof = iprof <= thr
        # initialization of the computation of the connected component
        edgeflag = False # True if interior is connected to the edges
        outflag = False # True if the propagation node is outside the interior
        interior = np.zeros_like(iprof, dtype=bool)
        nodes = tuple(np.array(node, dtype=int).reshape((3, 1)))
        for i in cycle((0, 1, 2)): # propagate through directions x, y, z
            if outflag: # node outside fprof
                dicho_low.append(thr)
                break
            elif edgeflag: # center connected to the edges: threshold too high
                dicho_high.append(thr)
                break
            elif nodes[0].size == 0: # filled interior: threshold low enough
                success = True
                dicho_low.append(thr)
                break
            edgeflag, outflag, nodes = _propagate(
                interior, fprof, nodes, axis=i)
    if not success:
        warnings.warn(
            "get_fill_threshold: interior threshold not found",
            RuntimeWarning)
    return success, dicho_low[-1], dicho_high[-1] - dicho_low[-1]


def get_connected_comp(fprof: np.ndarray,
                       node: tuple,
                       mask: np.ndarray = None)-> np.ndarray:
    """
    Compute the interior of the trap given a filled profile fprof.
    
    The interior is defined as the connected component of the center
    of the trap (ie the center of the array defining the trap).
    
    Note that the interior will also contain the "exterior" of the trap
    if the latter is connected to the center of the trap.

    Parameters
    ----------
    fprof : np.ndarray of bool
        The filled profile. Defined as iprof < threshold for given
        threshold and intensity profile iprof.
        The threshold must be suitably chosen such that the interior
        and exterior do not belong to the same connected component.
    mask : np.ndarray of bool, optional
        Data mask, fprof values are considered only where mask is True.

    Returns
    -------
    interior : np.ndarray of bool
        Same shape as fprof.
        interior[i1, i2, ..., in] is True if coordinate (i1, i2, ..., in)
        is located inside the trapping region

    """
    n = fprof.ndim
    interior = np.zeros_like(fprof, dtype=bool)
    nodes = tuple(np.array(node).reshape((n, 1)))
    for i in cycle(range(n)):
        if nodes[0].size == 0:
            return interior
        *_, nodes = _propagate(interior, fprof, nodes, axis=i)
    


# =============================================================================
# Escape path method
# =============================================================================

def get_escape_path_threshold(profSect: np.ndarray,
                              startPoint: tuple,
                              nbit: int = 13)-> float:
    """
    TODO split escape path and dichotomy
    Compute the intensity threshold defining the interior of the trap
    intensity profile section profSect, by trying to escape from it.
    
    This threshold is defined such that it is not possible to escape
    from the center of the trap without reaching a point of intensity
    > threshold.
    
    The function converges toward this threshold by dichotomy.
    
    It is recommended to use <get_fill_threshold> instead, as it get the
    threshold in 3d, hence a better management of edges.

    Parameters
    ----------
    profSect : 2D np.ndarray
        Section of the intensity profile. Care to take it in a plane
        where the trap is the weakest.
    startPoint : tuple (i0, j0) of int
        The starting point of the path. It should lie in the interior.
    nbit : int
        Number of dichotomy iterations.

    Returns
    -------
    float
        A lower bound on the confining threshold intensity of the trap.

    """
    dichotopath_high = [np.max(profSect)] # open trap thresholds
    dichotopath_low = [np.min(profSect)] # closed trap thresholds
    # Possible displacement directions
    direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
    # Dichotomy loop
    for i in range(nbit):
        # New threshold
        thr = (dichotopath_high[-1] + dichotopath_low[-1]) / 2
        prthr = profSect < thr
        curdir = 0 # Current direction index, start in direction (1, 0)
        
        path = np.zeros((profSect.size, 3), dtype=int)
        path[0, 0], path[0, 1] = startPoint[0], startPoint[1]
        stop = False # stop walking the path
        hitEdge = False # If an edge has been touched during the first part
        step = 0 # path step index
        # first part of the walk: seek an edge
        while not stop and not hitEdge:
            pos = path[step, :2]
            step += 1
            x, y = pos + direction[curdir]
            try:
                if prthr[x, y]: # still in the interior: continue straight on
                    path[step, 0], path[step, 1] = x, y
                    path[step, 2] = curdir
                else: # Touched an edge of the interior: go to second part
                    path[step, :2] = pos
                    curdir = (curdir + 1) % 4
                    hitEdge = True
            except IndexError: # hit upper corner: stop
                stop = True
                dichotopath_high.append(thr)
        # second part of the walk: try to escape
        while not stop:
            pos = path[step, :2]
            step += 1
            # try to go to the right of the current direction
            # then try the other directions counterclockwise
            for i in range(4):
                x, y = pos + direction[(curdir-1+i) % 4]
                try:
                    if x == -1 or y == -1: # hit lower corner: stop
                        stop = True
                        dichotopath_high.append(thr)
                    elif prthr[x, y]:
                        path[step, 0], path[step, 1] = x, y
                        curdir = (curdir - 1 + i) % 4
                        path[step, 2] = curdir
                        break
                except IndexError: # hit upper corner: stop
                    stop = True
                    dichotopath_high.append(thr)
            # if the path cycles: stop, threshold low enough
            for i in range(step):
                if np.all(path[step, :] == path[i, :]):
                    stop = True
                    dichotopath_low.append(thr)
                    break
    return dichotopath_low[-1]

