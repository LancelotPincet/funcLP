#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_shapes

"""
Defines data shape (nmodels, npoints) from inputs.
"""



# %% Libraries
import numpy as np



# %% Function
def use_shapes(variables, data, parameters) :
    '''
    Defines data shape (nmodels, npoints) from inputs.
    
    Parameters
    ----------
    variables : list
        List of variables arrays.
    data : list
        List of data arrays.
    parameters : list
        List of parameters arrays.

    Returns
    -------
    out_shape : tuple
        Output shape (nmodels, npoints).
    in_shapes : tuple of tuple
        (variables_shape, data_shape, parameters_shape) shapes of inputs
    '''

    # Broadcast shapes
    data_shape = np.broadcast_shapes(*[np.shape(arr) for arr in data]) if len(data) > 0 else (1, 1)
    parameters_shape = np.broadcast_shapes(data_shape[0], *[np.shape(arr) for arr in parameters]) if len(parameters) > 0 else tuple(data_shape[0])
    variables_shape = np.broadcast_shapes(data_shape[1:], *[np.shape(arr) for arr in variables]) if len(variables) > 0 else tuple(data_shape[1:])
    
    # Calculate outputs
    data_shape = parameters_shape + variables_shape
    nmodels = data_shape[0]
    npoints = np.prod(data_shape[1:])
    return (nmodels, npoints), (variables_shape, data_shape, parameters_shape)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)