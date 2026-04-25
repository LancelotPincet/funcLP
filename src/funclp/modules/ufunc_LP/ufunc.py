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
from funclp.modules.Function_LP._functions.Parameter import Parameter
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
    >>> from funclp import Parameter, ufunc
    >>> import numpy as np
    ... 
    >>> class MyClass() :
    ...     @ufunc(variables=['x', 'y'], data=['constant'], parameters=[Parameter('a', 1), Parameter('b', 0)])
    ...     def myfunc(x, y, constant, /, a, b) :
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

    def __init__(self, *, variables=None, data=None, parameters=None, constants=None, main=False, fastmath=True) :
        self._variables_metadata = variables
        self._data_metadata = data
        self._parameters_metadata = parameters
        self._constants_metadata = constants
        self.main = main
        self.fastmath = fastmath

    def __call__(self, function):
        '''Decorator logic'''
        self.function = function
        self.signature = inspect.signature(function)
        self._prepared = False

        return self

    def _prepare_metadata(self, owner=None, name=None):
        if self._prepared :
            return

        signature_parameters = self.signature.parameters

        inherited = None
        needs_inheritance = self._variables_metadata is None or self._parameters_metadata is None
        if needs_inheritance and owner is not None and name != 'function' :
            main_ufunc = getattr(owner, 'function', None)
            if isinstance(main_ufunc, ufunc) and main_ufunc is not self :
                main_ufunc._prepare_metadata(owner, 'function')
                inherited = main_ufunc

        if self._variables_metadata is None :
            if inherited is None :
                raise SyntaxError('@ufunc must define variables=[...] explicitly')
            self.variables = list(inherited.variables)
        else :
            self.variables = list(self._variables_metadata)

        self.data = list(inherited.data if self._data_metadata is None and inherited is not None else self._data_metadata or [])
        self.constants = list(inherited.constants if self._constants_metadata is None and inherited is not None else self._constants_metadata or [])

        if self._parameters_metadata is None :
            if inherited is None :
                raise SyntaxError('@ufunc must define parameters=[Parameter(...), ...] explicitly')
            self.parameters = list(inherited.parameters)
            self.parameter_specs = dict(inherited.parameter_specs)
        else :
            self.parameters, self.parameter_specs = self._normalize_parameters(self._parameters_metadata)

        declared_inputs = self.variables + self.data + self.parameters + self.constants
        for key in declared_inputs :
            if key not in signature_parameters :
                raise SyntaxError(f'{key} is declared in @ufunc metadata but is missing from the function signature')

        # Checking input definition follows rules 
        passed_data = False 
        for key, value in signature_parameters.items() :
            match value.kind :
                case inspect.Parameter.POSITIONAL_ONLY :
                    if key in self.data :
                        passed_data = True
                    elif key not in self.variables :
                        raise SyntaxError(f'{key} positional-only input must be declared in variables or data')
                    elif passed_data :
                        raise SyntaxError('Cannot define variables inputs after data inputs, please put all variables before data')
                case inspect.Parameter.POSITIONAL_OR_KEYWORD :
                    if key not in self.parameters and key not in self.constants :
                        raise SyntaxError(f'{key} parameter input must be declared with Parameter(...) or constants')
                case inspect.Parameter.KEYWORD_ONLY :
                    raise SyntaxError('ufunc cannot have keyword only attributes')

        self.inputs = ', '.join(declared_inputs)
        self.d_inputs = ', '.join([key for key in self.variables] + [key for key in self.data] + ['/'] + [key for key in self.parameters] + [key for key in self.constants])
        self.indexes_variables = ', '.join([f'{key}[point]' for key in self.variables]) + ', ' if len(self.variables) > 0 else ''
        self.indexes_data = ', '.join([f'{key}[model, point]' for key in self.data]) + ', ' if len(self.data) > 0 else ''
        self.indexes_parameters = ', '.join([f'{key}[model]' for key in self.parameters]) + ', ' if len(self.parameters) > 0 else ''
        self.indexes_constants = ', '.join([key for key in self.constants]) + ', ' if len(self.constants) > 0 else ''

        self._prepared = True

    def _normalize_parameters(self, metadata):
        if isinstance(metadata, dict) :
            metadata = metadata.values()

        names = []
        specs = {}
        for spec in metadata :
            if not isinstance(spec, Parameter) :
                raise SyntaxError('ufunc parameters must be declared with Parameter(...)')
            names.append(spec.name)
            specs[spec.name] = spec
        return names, specs

    def __get__(self, instance, owner):
        '''function called'''
        return getattr(instance, f'_{self.name}') if instance else self



    def __set_name__(self, cls, name):
        '''Descriptor setup'''
        self.name = name
        self._prepare_metadata(cls, name)
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
