#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-18
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : make_calculation

"""
Does the calculation for a given ufunc.
"""



# %% Libraries
from funclp import use_inputs, use_shapes, use_cuda, use_broadcasting



# %% Function
def make_calculation(function, name, args, kwargs, out) :
    '''
    Does the calculation for a given ufunc.
    
    Parameters
    ----------
    function : Function
        the function object where to calculate.
    name : str
        Name of the ufunc to calculate.
    args : tuple
        Argument to use for calculation.
    kwargs : dict
        Keyword arguments to use for calculation.
    out : xp.ndarray
        Output array, if None will initialize it automatically.

    Returns
    -------
    out : xp.ndarray
        output array.
    others : dict
        dictionnary containing all the other parameters calculated for this calculation.
        [nomodel, nopoint, nmodels, npoints, variables_shape, data_shape, parameters_shape, cuda, xp, transfer_back, blocks_per_grid, threads_per_block, variables, data, parameters, dtype, jitted]
    '''
    
    # Get ufunc
    ufunc = getattr(function.__class__, name)
    if ufunc is None :
        raise SyntaxError(f'ufunc {name} was not recognized')

    # Adds main parameters to kwargs
    ignore = kwargs.pop('ignore', False)
    if hasattr(function, "parameters") :
        function_parameters = {key: value for key, value in function.parameters.items() if key in ufunc.parameters}
        kwargs = {**function_parameters, **kwargs}

    # Use inputs
    inputs = use_inputs(ufunc, args, kwargs) # variables, data, parameters

    # Get shapes
    empty_dimensions, out_shape, in_shapes = use_shapes(*inputs) # (nomodel, nopoint), (nmodels, npoints), (variables_shapes, data_shapes, parameters_shapes)
    variables_shape, data_shape, parameters_shape = in_shapes
    nmodels, npoints = out_shape
    nomodel, nopoint = empty_dimensions

    # Define cuda
    cuda, xp, transfer_back, blocks_per_grid, threads_per_block = use_cuda(function, out_shape, inputs)

    # Broadcasting
    variables, data, parameters, dtype = use_broadcasting(xp, *inputs, *in_shapes, out_shape)
    ignore = xp.broadcast_to(xp.asarray(ignore).astype(xp.bool_), parameters_shape).reshape(nmodels)

    # Output
    out = xp.empty(shape=out_shape, dtype=dtype) if out is None else xp.asarray(out)
    if out.shape != out_shape :
        raise ValueError(f'Out shape {out.shape} is does not correspond to expected {out_shape}')

    # Function jitted
    jitted = getattr(function, f'gpu_{name}')[blocks_per_grid, threads_per_block] if cuda else getattr(function, f'cpu_{name}')
    
    # Apply function
    jitted(*variables, *data, *parameters, out, ignore)

    # Modify output
    out = out.reshape(data_shape)
    if transfer_back : out = xp.asnumpy(out)
    if nomodel : out = out[0, ...]
    if nopoint : out = out[..., 0]

    # Returns others
    others = dict(
        nomodel = nomodel,
        nopoint = nopoint,
        nmodels = nmodels,
        npoints = npoints,
        variables_shape = variables_shape,
        data_shape = data_shape,
        parameters_shape = parameters_shape,
        cuda = cuda,
        xp = xp,
        transfer_back = transfer_back,
        blocks_per_grid = blocks_per_grid,
        threads_per_block = threads_per_block,
        variables = variables,
        data = data,
        parameters = parameters,
        dtype = dtype,
        jitted = jitted,
    )
    return out, others



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)