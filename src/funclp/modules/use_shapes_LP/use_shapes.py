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
    (nomodel, nopoint) : tuple of bool
        True if the dimension of corresponding is 1 but should be removed at output
    (nmodels, npoints) : tuple of int
        Output shape.
    (variables_shape, data_shape, parameters_shape) : tuple of tuple
        Shapes of inputs
    '''

    # Broadcast shapes
    variables_shape = np.broadcast_shapes(*[np.shape(arr) for arr in variables]) if len(variables) > 0 else None
    data_shape = np.broadcast_shapes(*[np.shape(arr) for arr in data]) if len(data) > 0 else None
    parameters_shape = np.broadcast_shapes(*[np.shape(arr) for arr in parameters]) if len(parameters) > 0 else None

    # Check shapes for various scenarios
    match (variables_shape is not None and len(variables_shape) != 0, data_shape is not None and len(data_shape) != 0, parameters_shape is not None and len(parameters_shape) != 0):

        case (True, True, True):
            if data_shape[1:] != variables_shape or data_shape[:1] != parameters_shape :
                raise ValueError(f'Data shape {data_shape} does not correspond to parameters and variables shape {parameters_shape}, {variables_shape}')
            nomodel, nopoint = False, False

        case (True, False, True):
            nomodel = len(parameters_shape) == 0
            nopoint = len(variables_shape) == 0
            if nopoint and not nomodel :
                variables_shape, data_shape, parameters_shape = (1,), (*parameters_shape, 1), parameters_shape
            elif nomodel and not nopoint :
                variables_shape, data_shape, parameters_shape = variables_shape, (1, *variables_shape), (1,)
            else :
                variables_shape, data_shape, parameters_shape = variables_shape, (*parameters_shape, *variables_shape), parameters_shape
                # case nomodel and nopoint below

        case (False, True, True):
            if len(parameters_shape) == len(data_shape) :
                nomodel = len(parameters_shape) == 0
                nopoint = True
                variables_shape, data_shape, parameters_shape = (1,), (*data_shape, 1), parameters_shape
                for i in range(len(data)):
                    data[i] = data[i].reshape(*data[i].shape, 1)
                # case nomodel and nopoint below
            elif len(parameters_shape) < len(data_shape):
                nomodel = len(parameters_shape) == 0
                nopoint = False
                variables_shape, data_shape, parameters_shape = data_shape[1:], data_shape, parameters_shape
            else :
                raise ValueError(f'Parameters shape and data shape cannot be : {parameters_shape}, {data_shape}')

        case (True, True, False):
            if len(variables_shape) == len(data_shape) :
                nomodel = True
                nopoint = len(variables_shape) == 0
                variables_shape, data_shape, parameters_shape = variables_shape, (1, *data_shape), (1,)
                for i in range(len(data)):
                    data[i] = data[i].reshape(1, *data[i].shape)
                # case nomodel and nopoint below
            elif (len(variables_shape) + 1) == len(data_shape):
                nomodel = False
                nopoint = len(variables_shape) == 0
                variables_shape, data_shape, parameters_shape = variables_shape, data_shape, data_shape[:1]
            else :
                raise ValueError(f'Variables shape and data shape cannot be : {variables_shape}, {data_shape}')

        case (False, False, True):
            nomodel = len(parameters_shape) == 0
            nopoint = True
            variables_shape, data_shape, parameters_shape = (1,), (*parameters_shape, 1), parameters_shape
            # case nomodel and nopoint below

        case (True, False, False):
            nomodel = True
            nopoint = len(variables_shape) == 0
            variables_shape, data_shape, parameters_shape = variables_shape, (1, *variables_shape), (1,)
            # case nomodel and nopoint below

        case (False, True, False):
            nomodel, nopoint = False, False
            variables_shape, data_shape, parameters_shape = data_shape[1:], data_shape, data_shape[:1]

        case (False, False, False):
            if variables_shape is not None or data_shape is not None or parameters_shape is not None :
                nomodel, nopoint = True, True
            else :
                raise ValueError('Some inputs must be set for Function call')

    # Correct shape together
    if nomodel and nopoint :
        variables_shape, data_shape, parameters_shape = (1,), (1, 1), (1,)
    else :
        variables_shape = np.broadcast_shapes(variables_shape, data_shape[1:])
        parameters_shape = np.broadcast_shapes(parameters_shape, data_shape[:1])
        data_shape = parameters_shape + variables_shape
        if len(parameters_shape) > 1 :
            raise ValueError('Parameters cannot be on more than one dimension')    

    # Calculate outputs
    nmodels = data_shape[0]
    npoints = np.prod(data_shape[1:])
    return (nomodel, nopoint), (nmodels, npoints), (variables_shape, data_shape, parameters_shape)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)