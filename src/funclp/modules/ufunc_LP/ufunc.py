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
from funclp import use_inputs, use_shapes, use_cuda, use_broadcasting
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
    >>> class MyClass() :
    ...     @ufunc() # <-- HERE IS THE DECORATOR TO USE ON A SCALAR METHOD
    ...     def myfunc(x, y, /, constant, *, a, b) :
    ...         # Positional only : variables of size npoints, all broadcastable together
    ...         # Positional and keyword : data of same shape as output
    ...         # Keyword only : parameters of size nmodels, all broadcastable together
    ...         return a * x + b * y + constant
    ...     cuda = False
    ... 
    >>> instance = MyClass()
    ... 
    >>> x = np.arange(20).reshape((1, 20)) # Example of variables that can be broadcasted together
    >>> y = np.arange(20).reshape((20, 1)) # Example of variables that can be broadcasted together
    >>> constant = np.ones((5, 20, 20)) # Example of data with full shape
    >>> a = np.arange(5) # Example of parameter vector
    >>> b = 0.5 # Example of scalar use
    >>> cpu_out = instance.myfunc(x, y, constant, a=a, b=b)
    >>> instance.cuda = True
    >>> gpu_out = instance.myfunc(x, y, constant=constant, a=a, b=b)

    ...
    '''

    def __init__(self, *, main=False) :
        self.main = main

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

    def __get__(self, instance, owner):
        '''function called'''
        return getattr(instance, f'_{self.name}') if instance else self



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
        def cpu(instance):
            func = getattr(cls, f'_cpu_{name}', None)
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
                setattr(cls, f'_cpu_{name}', func)
            return func
        setattr(cls, f'cpu_{name}', cpu)

        @property
        def gpu(instance):
            func = getattr(cls, f'_gpu_{name}', None)
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
                setattr(cls, f'_gpu_{name}', func)
            return func
        setattr(cls, f'gpu_{name}', gpu)



        # Universal function

        def func(instance, *args, **kwargs):

            # Adds main parameters to kwargs
            if self.main : kwargs = {**instance.parameters, **kwargs}

            # Use inputs
            inputs = use_inputs(self, args, kwargs) # variables, data, parameters

            # Get shapes
            out_shape, in_shapes = use_shapes(*inputs) # (nmodels, npoints), (variables_shapes, data_shapes, parameters_shapes)

            # Define cuda
            cuda, xp, transfer_back, blocks_per_grid, threads_per_block = use_cuda(instance, out_shape, inputs)

            # Broadcasting
            variables, data, parameters, dtype = use_broadcasting(xp, *inputs, *in_shapes, out_shape)

            # Output
            out = xp.empty(shape=out_shape, dtype=dtype)

            # Function jitted
            jitted = getattr(instance, f'gpu_{name}')[blocks_per_grid, threads_per_block] if cuda else getattr(instance, f'cpu_{name}')
            
            # Apply function
            jitted(*variables, *data, *parameters, out)

            # Modify output
            out = out.reshape(in_shapes[1])
            if transfer_back : out = xp.asnumpy(out)
            return out

        setattr(cls, f'_{name}', func)



        # Main wrapper

        if not self.main : return # Stop if not a main function
        cls.variables, cls.data, cls.parameters = self.variables, self.data, self.parameters
        for pname, param in self.signature.parameters.items():
            if pname not in cls.parameters : continue

            #Single property
            @property
            def prop(self, pname=pname) :
                _param = getattr(self, f'_{pname}', None)
                if _param is not None :
                    return _param
                return getattr(self, f'_{pname}0')
            @prop.setter
            def prop(self, value, pname=pname) :
                if value is None :
                    setattr(self, f'_{pname}', None)
                else :
                    try :
                        value.astype(np.float32)
                    except AttributeError:
                        value = np.float32(value)
                    setattr(self, f'_{pname}', value)
            setattr(cls, pname, prop)

            #Default value
            if param.default is not inspect._empty :
                setattr(cls, f'_{pname}0', np.float32(param.default))
            else :
                raise SyntaxError(f'{pname} parameter should have a default value')

            #Estimate parameters function
            estimate = param.annotation
            if estimate is inspect._empty :
                setattr(cls, f'{pname}0', None)
                setattr(cls, f'{pname}_min', -np.float32(np.inf))
                setattr(cls, f'{pname}_max', +np.float32(np.inf))
                setattr(cls, f'{pname}_fit', False)
            else :
                setattr(cls, f'{pname}0', staticmethod(estimate))
                setattr(cls, f'{pname}_fit', True)
                minmax = estimate.__annotations__.get('return', None)
                mini, maxi = (None, None) if minmax is None else minmax
                if mini is None : mini = -np.float32(np.inf)
                if maxi is None : maxi = +np.float32(np.inf)
                setattr(cls, f'{pname}_min', mini)
                setattr(cls, f'{pname}_max', maxi)





# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)