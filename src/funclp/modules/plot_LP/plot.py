#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-18
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : plot

"""
Makes a plot of a function object.
"""



# %% Libraries
from funclp import make_calculation
from plotlp import figure
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None


# %% Function
def plot(function, path, args, kwargs={}, name=None, nmodel2plot=None) :
    '''
    Makes a plot of a function object.
    
    Parameters
    ----------
    function : Function
        Function object to plot on the data.
    path : str or pathlib.Path
        Path where to save the figure(s).
    args : tuple
        Argment to input in the function for calculating output.
    kwargs : dic
        Keyword argment to input in the function for calculating output.
    name : str
        Name of plot, if None will use name of function object.
    nmodel2plit : int
        Number of model to plot when having a multimodels data.

    Returns
    -------
    out : xp.ndarray
        Output array.

    Examples
    --------
    >>> from funclp import plot
    >>> from funclp.functions import Gaussian
    >>> import numpy as np
    ...
    >>> variables = (np.linspace(-1, 1, 1000),) # tuple
    >>> instance = Gaussian()
    >>> plot(instance, '', variables)
    '''

    name = function.name if name is None else name

    # Make calculation
    out, others = make_calculation(function, "function", args, kwargs)
    variables, variables_shape, xp, nomodel = others["variables"], others["variables_shape"], others["xp"], others["nomodel"]

    # Check CPU
    if xp is not np :
        cp = others["xp"]
        result = cp.asnumpy(out)
        variables = [cp.asnumpy(arr) for arr in variables]
    else :
        result = out

    # Check model stack
    if nomodel : result = result.reshape(1, *result.shape)
    indices = np.arange(len(result))
    if nmodel2plot is not None : np.random.shuffle(indices)
    else : nmodel2plot = len(result)

    # Check plot logic
    if len(variables) != len(variables_shape) :
        raise ValueError(f'To plot a function, number of variable ({len(variables)}) must correspond to number of output dimensions ({variables_shape})')

    # Loop
    for pos in range(nmodel2plot) :
        res = result[indices[pos]]
        ratio = np.shape(res)[1] / np.shape(res)[0] * 1.1 if res.ndim == 2 else None
        fig = figure(figsize_ratio=ratio)
        ax = fig.axis
        string = name if nomodel else f'{name}_[{indices[pos]:03}]'

        #Plot
        if res.ndim == 1 :
            x = variables[0]
            ax.plot(x, res)
            ax.set_xlabel(function.variables[0])
            ax.set_ylabel(f'{name} ({function.variables[0]})')
        elif res.ndim == 2 :
            x, y = variables[0].reshape(variables_shape)[0,:], variables[1].reshape(variables_shape)[:,0]
            barname = f'{name} ({function.variables[0]}, {function.variables[1]})'
            ax.imshow(res, coordinates=(x, y), barname=barname)
            ax.set_xlabel(function.variables[0])
            ax.set_ylabel(function.variables[1])
        else :
            raise TypeError(f'Cannot plot function apply with {res.ndim} dimensions')

        #Save
        fig.suptitle(string)
        fig.savefig((path / string).with_suffix('.png'))
        
    return out



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)