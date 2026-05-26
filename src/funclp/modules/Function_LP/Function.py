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
from funclp import CudaReference, kernel_caching 
from abc import ABC, abstractmethod
import numpy as np



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

        # Looping on constants
        for pname in main_ufunc.constants:
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

        # Looping on parameters
        for pname in main_ufunc.parameters:
            spec = main_ufunc.parameter_specs.get(pname, None)
            if spec is None:
                raise SyntaxError(f'{pname} parameter should be declared in @ufunc(parameters=...)')

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
                module_name = f'_{classname}_ufunc_{dname}'
                inputs_plus = main_ufunc.inputs.replace(pname, f'{pname} + eps')
                inputs_minus = main_ufunc.inputs.replace(pname, f'{pname} - eps')
                parameters = ', '.join([parameter_code(main_ufunc.parameter_specs[key]) for key in main_ufunc.parameters])
                string = f'''
from ._{classname}_cpukernel_function import _{classname}_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables={main_ufunc.variables!r}, data={main_ufunc.data!r}, parameters=[{parameters}], constants={main_ufunc.constants!r}, fastmath=False)
def {dname}({main_ufunc.d_inputs}):
    eps = 1e-3 * max(1.0, abs({pname}))
    f_plus = kernel({inputs_plus})
    f_minus = kernel({inputs_minus})
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel({main_ufunc.inputs})
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
'''
                d_ufunc = kernel_caching(module_name, string, object_name=dname)
                d_ufunc.__set_name__(cls, dname)
                setattr(cls, dname, d_ufunc)

            setattr(cls, f'_{pname}0', convert(spec.default))

            #Estimate parameters function
            estimate = spec.estimate
            if estimate is None :
                setattr(cls, f'{pname}0', None)
                setattr(cls, f'{pname}_min', -np.float32(np.inf))
                setattr(cls, f'{pname}_max', +np.float32(np.inf))
                setattr(cls, f'{pname}_fit', bool(spec.fit))
            else :
                setattr(cls, f'{pname}0', staticmethod(estimate))
                setattr(cls, f'{pname}_fit', bool(spec.fit))
                mini, maxi = spec.bounds
                if mini is None : mini = -np.float32(np.inf)
                if maxi is None : maxi = +np.float32(np.inf)
                setattr(cls, f'{pname}_min', mini)
                setattr(cls, f'{pname}_max', maxi)

    def _cpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return ""

    def _gpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return ""

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return model_params

    def _gpu_assembly_model_setup_source(self, block_params, parameters):
        return block_params

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return f'''
            mod = model_scalar({inputs_scalar})
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev'''

    def _gpu_assembly_model_eval_source(self, inputs_threads):
        return f'''
        mod = model_scalar({inputs_threads})
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '\n'.join([
            f'''            if bool2fit[{pos}]:
                jacob_local[count] = d_{key}({inputs_scalar})
                count += 1'''
            for pos, key in enumerate(parameters)
        ])

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return '\n'.join([
            f'''        if bool2fit[{pos}]:
            jacob_local[count] = d_{key}({inputs_threads})
            count += 1'''
            for pos, key in enumerate(parameters)
        ])

    def _cpu_assembly_source(self, estimator):
        function_name, estimator_name, distribution_name = self._assembly_cache_suffix(estimator)
        variables = [key for key in self.variables]
        data = [key for key in self.data]
        parameters = [key for key in self.parameters.keys()]
        constants = [key for key in self.constants]
        inputs = ', '.join(variables + data + parameters + constants)
        inputs_scalar = ', '.join(
            [f'point_{key}' for key in variables]
            + [f'point_{key}' for key in data]
            + [f'model_{key}' for key in parameters]
            + constants
        )
        point_variables = '\n            '.join([f'point_{key} = {key}[point]' for key in variables])
        point_data = '\n            '.join([f'point_{key} = {key}[model, point]' for key in data])
        model_params = '\n        '.join([f'model_{key} = {key}[model]' for key in parameters])
        model_setup = self._cpu_assembly_model_setup_source(model_params, parameters)
        point_setup = '\n            '.join([s for s in (point_variables, point_data) if s])
        model_eval = self._cpu_assembly_model_eval_source(inputs_scalar)
        derivatives = self._cpu_assembly_derivatives_source(parameters, inputs_scalar)

        imports = [
            "import numpy as np",
            "import numba as nb",
            (
                f"from ._{function_name}_cpukernel_function import "
                f"_{function_name}_cpukernel_function as model_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_cpukernel_deviance "
                f"import _{estimator_name}_{distribution_name}_cpukernel_deviance as deviance_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_cpukernel_loss "
                f"import _{estimator_name}_{distribution_name}_cpukernel_loss as loss_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_cpukernel_fisher "
                f"import _{estimator_name}_{distribution_name}_cpukernel_fisher as fisher_scalar"
            ),
        ]
        for key in parameters:
            imports.append(
                f"from ._{function_name}_cpukernel_d_{key} "
                f"import _{function_name}_cpukernel_d_{key} as d_{key}"
            )
        extra_imports = self._cpu_assembly_extra_imports_source(estimator, function_name, estimator_name, distribution_name, parameters)

        body = f'''
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _{function_name}_{estimator_name}_{distribution_name}_cpu_assembly(
    raw_data, {inputs}, weights, chi2, gradient, hessian, bool2fit, ignore
):
    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    for model in nb.prange(nmodels):
        if ignore[model]:
            continue

        chi_local = nb.float32(0.0)
        grad_local = np.zeros(MAX_PARAMS, dtype=np.float32)
        hess_local = np.zeros(NHESS, dtype=np.float32)
        jacob_local = np.empty(MAX_PARAMS, dtype=np.float32)

        {model_setup}

        for point in range(npoints):
            {point_setup}
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

{model_eval}

            count = 0
{derivatives}

            for p in range(nparams):
                Jp = jacob_local[p]
                grad_local[p] += Jp * los
                for q in range(p, nparams):
                    idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                    hess_local[idx] += Jp * jacob_local[q] * fis

        chi2[model] = chi_local
        for p in range(nparams):
            gradient[model, p] = grad_local[p]
        for p in range(nparams):
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                v = hess_local[idx]
                hessian[model, p, q] = v
                hessian[model, q, p] = v
'''
        source = '\n'.join(imports)
        if extra_imports and extra_imports.strip():
            source += '\n' + extra_imports.strip('\n')
        return source + '\n\n' + body

    def _gpu_assembly_source(self, estimator):
        function_name, estimator_name, distribution_name = self._assembly_cache_suffix(estimator)
        variables = [key for key in self.variables]
        data = [key for key in self.data]
        parameters = [key for key in self.parameters.keys()]
        constants = [key for key in self.constants]
        inputs = ', '.join(variables + data + parameters + constants)
        inputs_threads = ', '.join(
            [f'thread_{key}' for key in variables]
            + [f'thread_{key}' for key in data]
            + [f'block_{key}' for key in parameters]
            + constants
        )
        thread_variables = '\n        '.join([f'thread_{key} = {key}[point]' for key in variables])
        thread_data = '\n        '.join([f'thread_{key} = {key}[model, point]' for key in data])
        block_params = '\n    '.join([f'block_{key} = {key}[model]' for key in parameters])
        model_setup = self._gpu_assembly_model_setup_source(block_params, parameters)
        point_setup = '\n        '.join([s for s in (thread_variables, thread_data) if s])
        model_eval = self._gpu_assembly_model_eval_source(inputs_threads)
        derivatives = self._gpu_assembly_derivatives_source(parameters, inputs_threads)

        imports = [
            "import numba as nb",
            "from numba import cuda",
            (
                f"from ._{function_name}_gpukernel_function import "
                f"_{function_name}_gpukernel_function as model_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_gpukernel_deviance "
                f"import _{estimator_name}_{distribution_name}_gpukernel_deviance as deviance_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_gpukernel_loss "
                f"import _{estimator_name}_{distribution_name}_gpukernel_loss as loss_scalar"
            ),
            (
                f"from ._{estimator_name}_{distribution_name}_gpukernel_fisher "
                f"import _{estimator_name}_{distribution_name}_gpukernel_fisher as fisher_scalar"
            ),
        ]
        for key in parameters:
            imports.append(
                f"from ._{function_name}_gpukernel_d_{key} "
                f"import _{function_name}_gpukernel_d_{key} as d_{key}"
            )
        extra_imports = self._gpu_assembly_extra_imports_source(estimator, function_name, estimator_name, distribution_name, parameters)

        body = f'''
TPB = 128
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.cuda.jit(cache=True, fastmath=True)
def _{function_name}_{estimator_name}_{distribution_name}_gpu_assembly(
    raw_data, {inputs}, weights, chi2, gradient, hessian, bool2fit, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)
    grad_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)
    hess_local = nb.cuda.local.array(NHESS, nb.float32)
    jacob_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)

    for p in range(MAX_PARAMS):
        grad_local[p] = 0.0
    for idx in range(NHESS):
        hess_local[idx] = 0.0

    {model_setup}

    for point in range(tid, npoints, bdim):
        {point_setup}
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

{model_eval}

        count = 0
{derivatives}

        for p in range(nparams):
            Jp = jacob_local[p]
            grad_local[p] += Jp * los
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                hess_local[idx] += Jp * jacob_local[q] * fis

    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_grad = nb.cuda.shared.array((TPB, MAX_PARAMS), nb.float32)
    s_hess = nb.cuda.shared.array((TPB, NHESS), nb.float32)

    s_chi[tid] = chi_local
    for p in range(MAX_PARAMS):
        s_grad[tid, p] = grad_local[p]
    for idx in range(NHESS):
        s_hess[tid, idx] = hess_local[idx]

    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
            for p in range(nparams):
                s_grad[tid, p] += s_grad[tid + stride, p]
            for p in range(nparams):
                for q in range(p, nparams):
                    idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                    s_hess[tid, idx] += s_hess[tid + stride, idx]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        chi2[model] = s_chi[0]
        for p in range(nparams):
            gradient[model, p] = s_grad[0, p]
        for p in range(nparams):
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                v = s_hess[0, idx]
                hessian[model, p, q] = v
                hessian[model, q, p] = v
'''
        source = '\n'.join(imports)
        if extra_imports and extra_imports.strip():
            source += '\n' + extra_imports.strip('\n')
        return source + '\n\n' + body

    def cpu_assembly_kernel(self, estimator):
        function_name, estimator_name, distribution_name = self._assembly_cache_suffix(estimator)
        _ = estimator.cpukernel_deviance
        _ = estimator.cpukernel_loss
        _ = estimator.cpukernel_fisher
        module_name = f'_{function_name}_{estimator_name}_{distribution_name}_cpu_assembly'
        cache = getattr(self.__class__, '_cpu_assembly_cache', {})
        if module_name not in cache:
            source = self._cpu_assembly_source(estimator)
            cache[module_name] = kernel_caching(module_name, source, object_name=module_name)
            setattr(self.__class__, '_cpu_assembly_cache', cache)
        return cache[module_name]

    def gpu_assembly_kernel(self, estimator):
        function_name, estimator_name, distribution_name = self._assembly_cache_suffix(estimator)
        _ = estimator.gpukernel_deviance
        _ = estimator.gpukernel_loss
        _ = estimator.gpukernel_fisher
        module_name = f'_{function_name}_{estimator_name}_{distribution_name}_gpu_assembly'
        cache = getattr(self.__class__, '_gpu_assembly_cache', {})
        if module_name not in cache:
            source = self._gpu_assembly_source(estimator)
            cache[module_name] = kernel_caching(module_name, source, object_name=module_name)
            setattr(self.__class__, '_gpu_assembly_cache', cache)
        return cache[module_name]

    def get_assembly(self, estimator, cuda, nmodels):
        if not cuda:
            return self.cpu_assembly_kernel(estimator)
        threads_per_block = 128
        blocks_per_grid = nmodels
        return self.gpu_assembly_kernel(estimator)[blocks_per_grid, threads_per_block]

    def _assembly_cache_suffix(self, estimator):
        function_name = self.__class__.__name__
        estimator_name = estimator.__class__.__name__
        distribution = getattr(estimator, "distribution", None)
        distribution_name = distribution.__class__.__name__ if distribution is not None else "None"
        return function_name, estimator_name, distribution_name

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



def parameter_code(parameter) :
    return f'Parameter({parameter.name!r}, {value_code(parameter.default)})'



def value_code(value) :
    if isinstance(value, np.generic) :
        return repr(value.item())
    return repr(value)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)
