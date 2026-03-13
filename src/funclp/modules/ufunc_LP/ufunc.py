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

    def __init__(self, *, main=False, data=[], constants=[]) :
        self.main = main
        self.variable2data = data
        self.variable2constants = constants

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
        self.d_inputs = ', '.join([key for key in self.variables] + [key for key in self.data] + ['/'] + [key for key in self.parameters])
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
def func({self.inputs}, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel({self.indexes_variables}{self.indexes_data}{self.indexes_parameters}{self.indexes_constants})
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
def func({self.inputs}, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel({self.indexes_variables}{self.indexes_data}{self.indexes_parameters}{self.indexes_constants})
'''
                glob = {'nb': nb, 'kernel': getattr(instance, f'gpukernel_{name}')}
                loc = {}
                exec(string, glob, loc)
                func = loc['func']
                setattr(cls, f'_gpu_{name}', func)
            return func
        setattr(cls, f'gpu_{name}', gpu)



        # Universal function

        def func(instance, *args, out=None, ignore=False, **kwargs):
            return make_calculation(instance, name, args, kwargs, out, ignore)[0] # ignore others (index 1)
        setattr(cls, f'_{name}', func)



        # Main wrapper

        if not self.main : return # Stop if not a main function

        # Inputs
        cls.variables, cls.data, cls.constants = self.variables, self.data, self.constants
        @property
        def prop(instance) :
            return {parameter : getattr(instance, parameter) for parameter in self.parameters}
        @prop.setter
        def prop(instance, dic) :
            for key, value in dic.items() :
                setattr(instance, key, value)
        cls.parameters = prop
        @property
        def prop(instance) :
            return {constant : getattr(instance, constant) for constant in self.constants}
        @prop.setter
        def prop(instance, dic) :
            for key, value in dic.items() :
                setattr(instance, key, value)
        cls.constants = prop

        # Looping on parameters
        for pname, param in self.signature.parameters.items():

            #constant property
            if pname in self.constants :
                @property
                def prop(instance, pname=pname) :
                    _param = getattr(instance, f'_{pname}', None)
                    if _param is not None :
                        return _param
                    raise SyntaxeError('This constant should be defined at function init time')
                @prop.setter
                def prop(instance, value, pname=pname) :
                    setattr(instance, f'_{pname}', convert(value))
                setattr(cls, pname, prop)



            # Single parameters only below
            if pname not in self.parameters : continue

            #parameter property
            @property
            def prop(instance, pname=pname) :
                _param = getattr(instance, f'_{pname}', None)
                if _param is not None :
                    return _param
                return getattr(instance, f'_{pname}0')
            @prop.setter
            def prop(instance, value, pname=pname) :
                if value is None :
                    setattr(instance, f'_{pname}', None)
                else :
                    setattr(instance, f'_{pname}', convert(value))
            setattr(cls, pname, prop)

            # derivative property
            @property
            def prop(instance, pname=pname) :
                inputs_plus = self.inputs.replace(pname, f'{pname} + eps')
                inputs_minus = self.inputs.replace(pname, f'{pname} - eps')
                string = f'''
@ufunc(data={self.variable2data}, constants={self.variable2constants})
def d_param({self.d_inputs}, eps=1e-4) :
    f_plus, f_minus = kernel({inputs_plus}), kernel({inputs_minus})
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel({self.inputs})
    if np.isfinite(f_plus) and np.isfinite(f_x):
        return (f_plus - f_x) / eps
    if np.isfinite(f_minus) and np.isfinite(f_x):
        return (f_x - f_minus) / eps
    return np.nan
'''
                glob = {'ufunc': self.__class__, 'kernel': getattr(instance, f'cpukernel_{name}'), 'np': np}
                loc = {}
                exec(string, glob, loc)
                func = loc['d_param']
                setattr(cls, f'd_{pname}', func)
                return func
            setattr(cls, f'd_{pname}', prop)

            #Default value
            if param.default is not inspect._empty :
                setattr(cls, f'_{pname}0', convert(param.default))
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



def convert(value) :
    try :
        dtype = value.dtype
        if np.issubdtype(dtype, np.bool_) :
            return value.astype(np.bool_)
        elif np.issubdtype(dtype, np.integer) :
            return value.astype(np.int32)
        elif np.issubdtype(dtype, np.floating) :
            return value.astype(np.float32)
        else :
            raise TypeError(f'Parameter cannot have {dtype} dtype')
    except AttributeError:
        if isinstance(value, bool) or isinstance(value, np.bool_) :
            return np.bool_(value)
        elif isinstance(value, int) or isinstance(value, np.integer) :
            return np.int32(value)
        elif isinstance(value, float) or isinstance(value, np.floating) :
            return np.float32(value)
        else :
            raise TypeError(f'Parameter cannot have {type(value)} dtype')


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)