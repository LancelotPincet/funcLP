#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Function

"""
Function class defining a function model.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference
from abc import ABC, abstractmethod
import numpy as np
import inspect
import importlib
from pathlib import Path
cache_folder = Path(__file__).parents[1] / 'ufunc_LP/_functions'



# %% Class
class Function(ABC, CudaReference) :
    '''
    Function class defining a function model.
    
    Parameters
    ----------
    kwargs : dict
        Attributes to change.

    Attributes
    ----------
    parameters : dict
        dictionnary with eah parameter current value(s)
    nmodels : int
        Number of models according to current parameters of the function.
    '''

    @prop()
    def name(self) :
        return self.__class__.__name__
    def __init__(self, **kwargs) :
        selfkwargs(self, kwargs)
        for constant in self.constants.keys() :
            if not hasattr(self, constant) :
                raise SyntaxError(f'Please define {constant} constant to initialise this function')


    @abstractmethod
    def function(self) :
        pass
    def __call__(self, *args, **kwargs) :
        return self._function(*args, **kwargs)

    # Parameters
    @property
    def nmodels(self) :
        shape = np.broadcast_shapes(*[np.shape(getattr(self, param, [])) for param in self.parameters])
        if len(shape) > 1 :
            raise ValueError('Parameters cannot have more than 1 dimension')
        return shape[0] if len(shape) == 1 else 0

    # Set main ufunc parameters
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        main_ufunc = cls.function
        classname = cls.__name__
        
        # Inputs
        cls.variables, cls.data, cls.constants = main_ufunc.variables, main_ufunc.data, main_ufunc.constants
        @property
        def prop(instance) :
            return {parameter : getattr(instance, parameter) for parameter in main_ufunc.parameters}
        @prop.setter
        def prop(instance, dic) :
            for key, value in dic.items() :
                setattr(instance, key, value)
        cls.parameters = prop
        @property
        def prop(instance) :
            return {constant : getattr(instance, constant) for constant in main_ufunc.constants}
        @prop.setter
        def prop(instance, dic) :
            for key, value in dic.items() :
                setattr(instance, key, value)
        cls.constants = prop

        # Looping on parameters
        for pname, param in main_ufunc.signature.parameters.items():

            #constant property
            if pname in main_ufunc.constants :
                @property
                def prop(instance, pname=pname) :
                    _param = getattr(instance, f'_{pname}', None)
                    if _param is not None :
                        return _param
                    raise SyntaxError('This constant should be defined at function init time')
                @prop.setter
                def prop(instance, value, pname=pname) :
                    setattr(instance, f'_{pname}', convert(value))
                setattr(cls, pname, prop)

            # Single parameters only below
            if pname not in main_ufunc.parameters : continue

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

            # derivative default
            dname = f'd_{pname}'
            if dname not in cls.__dict__:
                file = cache_folder / f'_{classname}_ufunc_{dname}.py'
                inputs_plus = main_ufunc.inputs.replace(pname, f'{pname} + eps')
                inputs_minus = main_ufunc.inputs.replace(pname, f'{pname} - eps')
                string = f'''
from ._{classname}_cpukernel_function import _{classname}_cpukernel_function as kernel
from funclp import ufunc
import numpy as np
@ufunc(data={main_ufunc.variable2data}, constants={main_ufunc.variable2constants}, fastmath=False)
def {dname}({main_ufunc.d_inputs}, eps=1e-4):
    f_plus = kernel({inputs_plus})
    f_minus = kernel({inputs_minus})
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel({main_ufunc.inputs})
    if np.isfinite(f_plus) and np.isfinite(f_x):
        return (f_plus - f_x) / eps
    if np.isfinite(f_minus) and np.isfinite(f_x):
        return (f_x - f_minus) / eps
    return np.nan
'''
                if not file.exists() or file.read_text() != string:
                    file.write_text(string)
                importlib.invalidate_caches()
                module = importlib.import_module(f"funclp.modules.ufunc_LP._functions._{classname}_ufunc_{dname}")
                d_ufunc = getattr(module, dname)
                d_ufunc.__set_name__(cls, dname)
                setattr(cls, dname, d_ufunc)

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

        # CPU Jacobian
        file = cache_folder / f'_{classname}_cpu_jacobian.py'
        inputs = ', '.join(main_ufunc.variables + main_ufunc.data + main_ufunc.parameters + main_ufunc.constants)
        indexes = ', '.join([f'{key}[point]' for key in main_ufunc.variables] + [f'{key}[model, point]' for key in main_ufunc.data] + [f'{key}[model]' for key in main_ufunc.parameters] + [key for key in main_ufunc.constants])
        string = ''
        for pname in main_ufunc.parameters :
            string += f'from ._{classname}_cpukernel_d_{pname} import _{classname}_cpukernel_d_{pname} as d_{pname}\n'
        string += f'''
import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _{classname}_cpu_jacobian({inputs}, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0
'''
        for pos, pname in enumerate(main_ufunc.parameters) :
            string += f'''
            if bool2fit[{pos}] :
                jacobian[model, point, count] = d_{pname}({indexes})
                count += 1
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def cpu(instance):
            func = getattr(cls, f'_cpu_jacobian', None)
            if func is None:
                importlib.invalidate_caches()
                module = importlib.import_module(f"funclp.modules.ufunc_LP._functions._{classname}_cpu_jacobian")
                func = getattr(module, f"_{classname}_cpu_jacobian")
                setattr(cls, f'_cpu_jacobian', func)
            return func
        setattr(cls, f'cpu_jacobian', cpu)

        # GPU Jacobian
        file = cache_folder / f'_{classname}_gpu_jacobian.py'
        inputs = ', '.join(main_ufunc.variables + main_ufunc.data + main_ufunc.parameters + main_ufunc.constants)
        indexes = ', '.join([f'{key}[point]' for key in main_ufunc.variables] + [f'{key}[model, point]' for key in main_ufunc.data] + [f'{key}[model]' for key in main_ufunc.parameters] + [key for key in main_ufunc.constants])
        string = ''
        for pname in main_ufunc.parameters :
            string += f'from ._{classname}_gpukernel_d_{pname} import _{classname}_gpukernel_d_{pname} as d_{pname}\n'
        string += f'''
import numba as nb
from numba import cuda
@nb.cuda.jit()
def _{classname}_gpu_jacobian({inputs}, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0
'''
        for pos, pname in enumerate(main_ufunc.parameters) :
            string += f'''
        if bool2fit[{pos}] :
            jacobian[model, point, count] = d_{pname}({indexes})
            count += 1
'''
        if not file.exists() or file.read_text() != string:
            file.write_text(string)
        @property
        def gpu(instance):
            func = getattr(cls, f'_gpu_jacobian', None)
            if func is None:
                importlib.invalidate_caches()
                module = importlib.import_module(f"funclp.modules.ufunc_LP._functions._{classname}_gpu_jacobian")
                func = getattr(module, f"_{classname}_gpu_jacobian")
                setattr(cls, f'_gpu_jacobian', func)
            return func
        setattr(cls, f'gpu_jacobian', gpu)



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