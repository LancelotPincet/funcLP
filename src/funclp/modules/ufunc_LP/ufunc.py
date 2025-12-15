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



# %% Class
class ufunc() :
    '''
    Decorator class defining universal function factory object from python kernel function, will create kernels, vectorized functions, jitted functions, stack functions, all on CPU / Parallel CPU / GPU.
    
    Examples
    --------
    >>> from funclp import ufunc
    >>> import numpy as np
    ...
    >>> class MyClass() :
    ...     @ufunc() # <-- HERE IS THE DECORATOR TO USE
    ...     def myfunc(x, /, *, a, b) :
    ...         return a * x + b
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
        self.indexes = ', '.join([f'{key}[model, point]' for key in parameters.keys()])
  
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
            out[model, point] = kernel({self.indexes})
'''
                glob = {'nb': nb, 'kernel': getattr(cls, f'cpukernel_{name}')}
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
        newout[model, point] = kernel({self.indexes})
'''
                glob = {'nb': nb, 'kernel': getattr(cls, f'gpukernel_{name}')}
                loc = {}
                exec(string, glob, loc)
                func = loc['func']
                setattr(cls, f'_gpujit_{name}', func)
            return func
        setattr(cls, f'gpujit_{name}', gpujit)



        # Universal functions

        def cpu(instance, *args, **kwargs):
            variables, data, parameters = self.variables_data_parameters(instance, *args, **kwargs) # Get various arrays from inputs
            nmodels = self.parameters2nmodels(parameters)
            jitted = getattr(cls, f'cpujit_{name}')
            return send_to_jitted(*args, jitted, )
        setattr(cls, f'cpu_{name}', cpu)

        def gpu(instance):
            pass # TODO
        setattr(cls, f'cpu_{name}', gpu)



    # Helpers

    def variables_data_parameters(self, instance, *args, **kwargs) :
        '''This function separates variables, data and parameters from input values'''
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        variables = [bound.arguments[key] for key in self.variables]
        data = [bound.arguments[key] for key in self.data]
        parameters = [bound.arguments[key] for key in self.parameters]
        return variables, data, parameters
    
    def parameters2nmodels(self, parameters, data) :
        '''This function calculates the number of models'''
        data_shape = np.broadcast_shapes(*[datum.shape for datum in data]) if len(data) > 0 else None
        nmodels = np.broadcast_shapes(*[parameter.shape for parameter in parameters])[0] if len(parameters) > 0 else data_shape[0]
        
        # Error
        if data_shape is not None and data_shape[0] != nmodels :
            raise ValueError('data shape does not match with parameters')
        return nmodels

    def variables2npoints(self, variables, data) :
        '''This function calculates the number of points'''
        data_shape = np.broadcast_shapes(*[datum.shape for datum in data]) if len(data) > 0 else None
        shape = np.broadcast_shapes(*[variable.shape for variable in variables]) if len(parameters) > 0 else data_shape[0]
        
        # Error
        if data_shape is not None and data_shape[0] != nmodels :
            raise ValueError('data shape does not match with parameters')
        return nmodels

        

    # Call
    def __get__(self, instance, owner):
        '''function called'''
        if instance :
            if instance.cuda :
                return getattr(instance, f'gpu_{self.name}')
            return getattr(instance, f'cpu_{self.name}')
        return self



# Pass arrays CPU <> GPU
def to_gpu(arr):
    if isinstance(arr, np.ndarray): # 1. Already a NumPy array (CPU)
        return cuda.to_device(arr, stream=stream)
    if isinstance(arr, cuda.cudadrv.devicearray.DeviceNDArray): # 2. Already a Numba device array
        return arr

    # 3. Other CUDA-aware arrays (CuPy, etc.)
    if hasattr(arr, "__cuda_array_interface__"):
        return cuda.as_cuda_array(arr)

    # 4. Fallback: convert to NumPy then copy
    return cuda.to_device(np.asarray(arr), stream=stream)
def to_cpu(arr):
    if isinstance(arr, np.ndarray): # Case 1: Already a NumPy array
        return arr
    if isinstance(arr, cuda.cudadrv.devicearray.DeviceNDArray): # Case 2: Numba DeviceNDArray
        return arr.copy_to_host()
    if hasattr(arr, "__cuda_array_interface__"): # Case 3: Other CUDA arrays (CuPy, etc.)
        return cuda.as_cuda_array(arr).copy_to_host()
    return np.asarray(arr) # Case 4: Fallback (Python sequence, scalar, etc.)

# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)