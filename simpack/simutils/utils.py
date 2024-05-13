# -*- coding: utf-8 -*-
"""
General utility functions for the simpack.
    - <unpack_param>, to unpack gaussian beam parameters.

"""

from numbers import Number

import numpy as np

ParamType = tuple | list | np.ndarray | Number

# =============================================================================
# Utility functions
# =============================================================================

def unpack_param(param: ParamType):
    """
    Unpack gaussian beam parameter.
    """
    if isinstance(param, (tuple, list, np.ndarray)):
        return param[0], param[1]
    elif isinstance(param, Number):
        return float(param), float(param)
    else:
        raise TypeError(
            "incorrect variable type: expected tuple, list, array or number, "
            f"got {type(param)}")
