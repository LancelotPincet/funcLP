#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_broadcasting

"""
Defines inputs after broadcasting ready to use.
"""



# %% Libraries




# %% Function
def use_broadcasting(xp, variables, data, parameters, variables_shape, data_shape, parameters_shape, out_shape) :
    '''
    Defines inputs after broadcasting ready to use.
    
    Parameters
    ----------
    xp : module
        numpy / cupy.
    variables : list
        List of input variables.
    data : list
        List of input data.
    parameters : list
        List of input parameters.
    variables_shape : tuple
        Shape to apply to variables.
    data_shape : tuple
        Shape to apply to data.
    parameters_shape : tuple
        Shape to apply to parameters.
    out_shape : tuple
        Output shape.

    Returns
    -------
    variables : list
        List of broadcasted variables.
    data : list
        List of broadcasted data.
    parameters : list
        List of broadcasted parameters.
    '''

    # Data type
    dtype = xp.result_type(*variables, *data, *parameters)

    # Broadcasting
    nmodels, npoints = out_shape
    variables = [xp.broadcast_to(xp.asarray(arr).astype(dtype), variables_shape).reshape(npoints) for arr in variables]
    data = [xp.broadcast_to(xp.asarray(arr).astype(dtype), data_shape).reshape((nmodels, npoints)) for arr in data]
    parameters = [xp.broadcast_to(xp.asarray(arr).astype(dtype), parameters_shape).reshape(nmodels) for arr in parameters]

    return variables, data, parameters, dtype



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)