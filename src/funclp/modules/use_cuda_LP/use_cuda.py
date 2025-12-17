#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_cuda

"""
Defines cuda parameters.
"""



# %% Libraries
import numpy as np
import numba as nb
from numba import cuda
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def use_cuda(instance, out_shape, inputs) :
    '''
    Defines cuda parameters.
    
    Parameters
    ----------
    instance : Function
        function object.
    out_shape : tuple
        (nmodels, npoints) shape of output.
    inputs : tuple
        (variables, data, parameters) inputs.

    Returns
    -------
    cuda : bool
        True if dusing cuda.
    xp : module
        numpy / cupy
    transfer_back : bool
        True if cuda and all inputs where on CPU --> transfer back to CPU at the end
    blocks_per_grid : tuple
        blocks per grid for cuda kernel 2D (nmodels, npoints)
    threads_per_block : tuple
        threads per block for cuda kernel 2D (nmodels, npoints)
    '''

    nmodels, npoints = out_shape

    # Define cuda
    cuda = False if (cp is None or not nb.cuda.is_available()) else getattr(instance, "cuda", None)
    if cuda is None :
        cuda = instance.cpu2gpu < 4 * nmodels * npoints

    # xp
    xp = cp if cuda else np
    
    # Was on GPU
    transfer_back = not(any([isinstance(arr, cp.ndarray) for arr in sum(inputs, [])])) if cuda else False

    # Calculating kernel
    threads_per_block = 16, 16
    blocks_per_grid = (nmodels + threads_per_block[0] - 1) // threads_per_block[0], (npoints + threads_per_block[1] - 1) // threads_per_block[1]

    return cuda, xp, transfer_back, blocks_per_grid, threads_per_block



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)