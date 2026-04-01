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
from funclp import make_calculation
import inspect
import importlib
from pathlib import Path
cache_folder = Path(__file__).parent / '_functions'


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
    ...     def myfunc(x, y, constant, /, a, b) :
    ...         # Positional only : variables of size npoints, all broadcastable together, then all data of same shape as output, all broadcastable together (must be stepcified in data=[])
    ...         # Positional and keyword : parameters of size nmodels, all broadcastable together
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
    >>> gpu_out = instance.myfunc(x, y, constant, a=a, b=b)

    ...
    '''

    main_functions = {}

    def __init__(self, *, data=[], constants=[], fastmath=True) :
        self.variable2data = data
        self.variable2constants = constants
        self.fastmath = fastmath

    def __call__(self, function):
        '''Decorator logic'''
        self.function = function
        self.signature = inspect.signature(function)

        # Checking input definition follows rules 
        parameters = self.signature.parameters
        passed_data = False 
        for key, value in parameters.items() :
            match value.kind :
                case inspect.Parameter.POSITIONAL_ONLY :
                    if key in self.variable2data :
                        passed_data = True
                    elif passed_data :
                        raise SyntaxError('Cannot define variables inputs after data inputs, please put all variables before data')
                    else :
                        pass
                case inspect.Parameter.POSITIONAL_OR_KEYWORD :
                    continue
                case inspect.Parameter.KEYWORD_ONLY :
                    raise SyntaxError('ufunc cannot have keyword only attributes')

        # Get variable and parameters of the function 
        self.variables = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_ONLY and key not in self.variable2data]
        self.data = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_ONLY and key in self.variable2data]
        self.parameters = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and key not in self.variable2constants]
        self.constants = [key for key, value in parameters.items() if value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and key in self.variable2constants]
        self.inputs = ', '.join([key for key in parameters.keys()])
        self.d_inputs = ', '.join([key for key in self.variables] + [key for key in self.data] + ['/'] + [key for key in self.parameters] + [key for key in self.constants])
        self.indexes_variables = ', '.join([f'{key}[point]' for key in self.variables]) + ', ' if len(self.variables) > 0 else ''
        self.indexes_data = ', '.join([f'{key}[model, point]' for key in self.data]) + ', ' if len(self.data) > 0 else ''
        self.indexes_parameters = ', '.join([f'{key}[model]' for key in self.parameters]) + ', ' if len(self.parameters) > 0 else ''
        self.indexes_constants = ', '.join([key for key in self.constants]) + ', ' if len(self.constants) > 0 else ''
  
        return self

    def __get__(self, instance, owner):
        '''function called'''
        return getattr(instance, f'_{self.name}') if instance else self



    def __set_name__(self, cls, name):
        '''Descriptor setup'''
        self.name = name
        classname = cls.__name__
        self.main_functions[f'{classname}_{name}'] = self.function
        setattr(cls, f'python_{name}', self.function)

        # Kernels

        file = cache_folder / f'_{classname}_cpukernel_{name}.py'
        string = f'''
from funclp import ufunc
import numba as nb
_{classname}_cpukernel_{name} = nb.njit(nogil=True, cache=True)(ufunc.main_functions["{classname}_{name}"])
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def cpukernel(instance):
            func = getattr(cls, f'_cpukernel_{name}', None)
            if func is None:
                importlib.invalidate_caches()
                module = importlib.import_module(f"{__package__}._functions._{classname}_cpukernel_{name}")
                func = getattr(module, f"_{classname}_cpukernel_{name}")
                setattr(cls, f'_cpukernel_{name}', func)
            return func
        setattr(cls, f'cpukernel_{name}', cpukernel)

        file = cache_folder / f'_{classname}_gpukernel_{name}.py'
        string = f'''
from funclp import ufunc
import numba as nb
from numba import cuda
_{classname}_gpukernel_{name} = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["{classname}_{name}"])
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def gpukernel(instance):
            func = getattr(cls, f'_gpukernel_{name}', None)
            if func is None:
                importlib.invalidate_caches()
                module = importlib.import_module(f"{__package__}._functions._{classname}_gpukernel_{name}")
                func = getattr(module, f"_{classname}_gpukernel_{name}")
                setattr(cls, f'_gpukernel_{name}', func)
            return func
        setattr(cls, f'gpukernel_{name}', gpukernel)



        # Jitted functions

        file = cache_folder / f'_{classname}_cpu_{name}.py'
        string = f'''
from ._{classname}_cpukernel_{name} import _{classname}_cpukernel_{name} as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath={self.fastmath}, parallel=True)
def _{classname}_cpu_{name}({self.inputs}, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel({self.indexes_variables}{self.indexes_data}{self.indexes_parameters}{self.indexes_constants})
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def cpu(instance):
            func = getattr(cls, f'_cpu_{name}', None)
            if func is None:
                importlib.invalidate_caches()
                module = importlib.import_module(f"{__package__}._functions._{classname}_cpu_{name}")
                func = getattr(module, f"_{classname}_cpu_{name}")
                setattr(cls, f'_cpu_{name}', func)
            return func
        setattr(cls, f'cpu_{name}', cpu)

        file = cache_folder / f'_{classname}_gpu_{name}.py'
        string = f'''
from ._{classname}_gpukernel_{name} import _{classname}_gpukernel_{name} as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath={self.fastmath})
def _{classname}_gpu_{name}({self.inputs}, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel({self.indexes_variables}{self.indexes_data}{self.indexes_parameters}{self.indexes_constants})
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def gpu(instance):
            func = getattr(cls, f'_gpu_{name}', None)
            if func is None:
                module = importlib.import_module(f"{__package__}._functions._{classname}_gpu_{name}")
                func = getattr(module, f"_{classname}_gpu_{name}")
                setattr(cls, f'_gpu_{name}', func)
            return func
        setattr(cls, f'gpu_{name}', gpu)



        # Universal function

        def func(instance, *args, out=None, ignore=False, **kwargs):
            return make_calculation(instance, name, args, kwargs, out, ignore)[0] # ignore others (index 1)
        setattr(cls, f'_{name}', func)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)