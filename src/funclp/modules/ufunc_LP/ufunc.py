#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-13
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : ufunc

"""
Decorator class defining universal function factory object from python kernel function, will create kernels, vectorized functions, jitted functions, stack functions, all on CPU / Parallel CPU / GPU.
"""



# %% Libraries
from corelp import selfkwargs
import inspect
import numpy as np
import numba as nb
from numba import cuda
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Class
class ufunc() :
    '''
    Decorator class defining universal function factory object from python kernel function, will create kernels, vectorized functions, jitted functions, stack functions, all on CPU / Parallel CPU / GPU.
    
    Examples
    --------
    >>> from funclp import ufunc
    >>> import numpy as np
    ...
    '''

    def __init__(self, **kwargs) :
        selfkwargs(self, kwargs)

    def __call__(self, function):
        '''Decorator logic'''
        self.function = function
        self.signature = inspect.signature(function)

        # Get variable and parameters of the function 
        parameters = self.signature.parameters
        self.variables = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_ONLY]
        self.data = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        self.parameters = [key for key, value in parameters.items() if value.kind == inspect.Parameter.KEYWORD_ONLY]
        self.inputs = ', '.join([key for key in parameters.keys()])
        self.indexes_variables = ', '.join([f'{key}[point]' for key in self.variables])
        self.indexes_data = ', '.join([f'{key}[model, point]' for key in self.data])
        self.indexes_parameters = ', '.join([f'{key}[model]' for key in self.parameters])
  
        return self



    def __set_name__(self, cls, name):
        '''Descriptor setup'''
        self.name = name

        # Kernels

        @property
        def cpukernel(instance):
            func = getattr(cls, f'_cpukernel_{name}', None)
            if func is None:
                func = nb.njit(nogil=True)(self.function)
                setattr(cls, f'_cpukernel_{name}', func)
            return func
        setattr(cls, f'cpukernel_{name}', cpukernel)

        @property
        def gpukernel(instance):
            func = getattr(cls, f'_gpukernel_{name}', None)
            if func is None:
                func = nb.cuda.jit(device=True)(self.function)
                setattr(cls, f'_gpukernel_{name}', func)
            return func
        setattr(cls, f'gpukernel_{name}', gpukernel)



        # Jitted functions

        @property
        def cpujit(instance):
            func = getattr(cls, f'_cpujit_{name}', None)
            if func is None:
                string = f'''
@nb.njit()
def func({self.inputs}, out) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        for point in range(npoints) :
            out[model, point] = kernel({self.indexes_variables}, {self.indexes_data}, {self.indexes_parameters})
'''
                glob = {'nb': nb, 'kernel': getattr(instance, f'cpukernel_{name}')}
                loc = {}
                exec(string, glob, loc)
                func = loc['func']
                setattr(cls, f'_cpujit_{name}', func)
            return func
        setattr(cls, f'cpujit_{name}', cpujit)

        @property
        def gpujit(instance):
            func = getattr(cls, f'_gpujit_{name}', None)
            if func is None:
                string = f'''
@nb.cuda.jit()
def func({self.inputs}, out) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints :
        out[model, point] = kernel({self.indexes_variables}, {self.indexes_data}, {self.indexes_parameters})
'''
                glob = {'nb': nb, 'kernel': getattr(instance, f'gpukernel_{name}')}
                loc = {}
                exec(string, glob, loc)
                func = loc['func']
                setattr(cls, f'_gpujit_{name}', func)
            return func
        setattr(cls, f'gpujit_{name}', gpujit)



        # Universal functions

        def cpu(instance, *args, **kwargs):
            variables, data, parameters, out, data_shape, _, _ = self.use_inputs(*args, cuda=False, **kwargs)
            jitted = getattr(instance, f'cpujit_{name}')
            jitted(*variables, *data, *parameters, out)
            out = out.reshape(data_shape)
            return out
        setattr(cls, f'cpu_{name}', cpu)

        def gpu(instance, *args, **kwargs):
            variables, data, parameters, out, data_shape, was_on_gpu, (blocks_per_grid, threads_per_block) = self.use_inputs(*args, cuda=True, **kwargs)
            jitted = getattr(instance, f'gpujit_{name}')[blocks_per_grid, threads_per_block]
            jitted(*variables, *data, *parameters, out)
            out = out.reshape(data_shape)
            return cp.asarray(out) if was_on_gpu else cp.asnumpy(out)
        setattr(cls, f'gpu_{name}', gpu)



    # Helpers

    def use_inputs(self, *args, cuda=False, **kwargs) :
        '''This function converts inputs into all the needed parameters for the jitted functions'''

        # Checking Cuda
        was_on_gpu = any([isinstance(arr, cp.ndarray) for arr in list(args) + list(kwargs.values())]) if cuda else False
        asarray = np.asarray if cp is None else cp.asarray if cuda else cp.asnumpy
        xp = cp if cuda else np
        
        # Get dtype
        dtype = xp.result_type(*args, *kwargs.values())

        # Separate inputs
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        variables = [asarray(bound.arguments[key]).astype(dtype) for key in self.variables]
        data = [asarray(bound.arguments[key]).astype(dtype) for key in self.data]
        parameters = [asarray(bound.arguments[key]).astype(dtype) for key in self.parameters]
    
        # Get shapes
        variable_shapes = {arr.shape for arr in variables if arr.size != 1}
        data_shapes = {arr.shape for arr in data if arr.size != 1}
        parameters_shapes = {arr.shape for arr in parameters if arr.size != 1}
        if len(variable_shapes) > 1:
            raise ValueError(f"Inconsistent variable shapes: {variable_shapes}")
        if len(data_shapes) > 1:
            raise ValueError(f"Inconsistent data shapes: {data_shapes}")
        if len(parameters_shapes) > 1:
            raise ValueError(f"Inconsistent parameters shapes: {parameters_shapes}")
        data_shape = next(iter(data_shapes), (1, 1))
        variable_shape = next(iter(variable_shapes), data_shape[1:])
        parameters_shape = next(iter(parameters_shapes), (data_shape[0],) )
    
        # Get npoints and nmodels
        nmodels = parameters_shape[0]
        npoints = np.prod(variable_shape)
        if data_shape[0] != 1 and (data_shape[0],) != parameters_shape :
            raise ValueError(f"Inconsistent data / parameters shape compatibility: {data_shapes} / {parameters_shape}")
        if np.prod(data_shape[1:]) != 1 and data_shape[1:] != variable_shape :
            raise ValueError(f"Inconsistent data / variables shape compatibility: {data_shapes} / {variable_shape}")
        data_shape = parameters_shape + variable_shape

        # Arrays
        variables = [xp.ascontiguousarray(arr.reshape(npoints)) if arr.size > 1 else xp.full(npoints, arr) for arr in variables]
        data = [xp.ascontiguousarray(arr.reshape((nmodels, npoints))) if arr.size > 1 else xp.full((nmodels, npoints), arr) for arr in data]
        parameters = [xp.ascontiguousarray(arr.reshape(nmodels)) if arr.size > 1 else xp.full(nmodels, arr) for arr in parameters]
        out = xp.empty(shape=(nmodels, npoints), dtype=dtype) # output shape

        # GPU specifications
        threads_per_block = 16, 16
        blocks_per_grid = (nmodels + threads_per_block[0] - 1) // threads_per_block[0], (npoints + threads_per_block[1] - 1) // threads_per_block[1]

        return variables, data, parameters, out, data_shape, was_on_gpu, (blocks_per_grid, threads_per_block)



    # Call
    def __get__(self, instance, owner):
        '''function called'''
        if instance :
            if instance.cuda :
                return getattr(instance, f'gpu_{self.name}')
            return getattr(instance, f'cpu_{self.name}')
        return self


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)