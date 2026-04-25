#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference, Parameter
from funclp.modules.Function_LP.Function import convert
import inspect
import numpy as np
import numba as nb



# %% Channel definition
class JointChannel:
    '''Single channel definition for JointFunction.'''

    def __init__(self, function, variables=None, affine=None, offset=None, prefix=None):
        self.function = function
        self.variables = variables
        self.prefix = prefix
        self.affine = np.eye(2, dtype=np.float32) if affine is None else np.asarray(affine, dtype=np.float32)
        if self.affine.shape != (2, 2):
            raise ValueError('affine must have shape (2, 2)')
        self.offset = np.zeros(2, dtype=np.float32) if offset is None else np.asarray(offset)
        if len(self.offset) != 2:
            raise ValueError('offset must have length 2')



class _FitUfunc:
    def __init__(self, variables, parameters, parameter_specs):
        self.variables = variables
        self.data = []
        self.parameters = parameters
        self.parameter_specs = parameter_specs
        signature_parameters = [inspect.Parameter(key, inspect.Parameter.POSITIONAL_ONLY) for key in variables]
        signature_parameters += [inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=parameter_specs[key].default) for key in parameters]
        self.signature = inspect.Signature(signature_parameters)



# %% Joint function
class JointFunction(CudaReference):
    '''Composite function used to jointly fit several optical channels.'''

    @prop()
    def name(self):
        return self.__class__.__name__

    def __init__(self, channels, shared=('mux', 'muy'), **kwargs):
        self.channels = [channel if isinstance(channel, JointChannel) else JointChannel(channel) for channel in channels]
        if len(self.channels) < 2:
            raise ValueError('JointFunction needs at least two channels')
        self.shared = tuple(shared)
        if len(self.shared) != 2:
            raise ValueError('shared must contain two parameter names')
        self._setup_channels()
        self._setup_parameters()
        self._setup_constants()
        self._setup_fit_ufunc()
        self._build_kernels()
        selfkwargs(self, kwargs)
        self._sync_channels()

    def _setup_channels(self):
        variables = list(self.channels[0].function.variables)
        if len(variables) != 2:
            raise ValueError('JointFunction currently supports 2D functions only')
        for pos, channel in enumerate(self.channels):
            channel.prefix = channel.prefix or f'ch{pos}'
            if not channel.prefix.isidentifier():
                raise ValueError(f'{channel.prefix} is not a valid parameter prefix')
            if list(channel.function.variables) != variables:
                raise ValueError('All joint channels must use the same variables')
            if len(channel.function.data) != 0:
                raise ValueError('JointFunction does not support channel functions with data inputs')
            for pname in self.shared:
                if pname not in channel.function.parameters:
                    raise ValueError(f'Shared parameter {pname} is missing from channel {pos}')
        self.variables = ['channel', *variables]
        self.data = []

    def _setup_parameters(self):
        names = []
        specs = {}

        for pname in self.shared:
            source = self._shared_source(pname)
            specs[pname] = Parameter(
                pname,
                source.default,
                estimate=source.estimate,
                bounds=source.bounds,
                fit=source.fit,
            )
            names.append(pname)

        self._channel_parameter_names = []
        for channel in self.channels:
            channel_names = {}
            for pname, value in channel.function.parameters.items():
                if pname in self.shared:
                    continue
                name = f'{channel.prefix}_{pname}'
                spec = channel.function.__class__.function.parameter_specs[pname]
                specs[name] = Parameter(
                    name,
                    value,
                    estimate=spec.estimate,
                    bounds=spec.bounds,
                    fit=getattr(channel.function, f'{pname}_fit'),
                )
                names.append(name)
                channel_names[pname] = name
            self._channel_parameter_names.append(channel_names)

        self._transform_parameter_names = []
        for channel in self.channels:
            affine = channel.affine
            offset = channel.offset
            transform = {
                'xx': affine[0, 0], 'xy': affine[0, 1], 'x0': offset[0],
                'yx': affine[1, 0], 'yy': affine[1, 1], 'y0': offset[1],
            }
            transform_names = {}
            for suffix, value in transform.items():
                name = f'{channel.prefix}_{suffix}'
                specs[name] = Parameter(name, value, fit=False)
                names.append(name)
                transform_names[suffix] = name
            self._transform_parameter_names.append(transform_names)

        self._parameters_names = names
        self._parameter_specs = specs
        for key in self._parameters_names:
            setattr(self, key, convert(specs[key].default))
            setattr(self, f'{key}_min', -np.float32(np.inf) if specs[key].bounds[0] is None else specs[key].bounds[0])
            setattr(self, f'{key}_max', +np.float32(np.inf) if specs[key].bounds[1] is None else specs[key].bounds[1])
            setattr(self, f'{key}_fit', bool(specs[key].fit))

    def _shared_source(self, pname):
        for channel in self.channels:
            if pname in channel.function.__class__.function.parameter_specs:
                spec = channel.function.__class__.function.parameter_specs[pname]
                return Parameter(
                    pname,
                    channel.function.parameters[pname],
                    estimate=spec.estimate,
                    bounds=spec.bounds,
                    fit=getattr(channel.function, f'{pname}_fit'),
                )
        raise ValueError(f'Shared parameter {pname} was not found in any channel')

    def _setup_constants(self):
        self._constant_names = []
        self._constant_values = {}
        self._channel_constant_names = []
        for channel in self.channels:
            channel_names = {}
            for cname, value in channel.function.constants.items():
                name = f'{channel.prefix}_{cname}'
                self._constant_names.append(name)
                self._constant_values[name] = value
                channel_names[cname] = name
            self._channel_constant_names.append(channel_names)

    def _setup_fit_ufunc(self):
        self.fit_ufunc = _FitUfunc(self.variables, self._parameters_names, self._parameter_specs)

    @property
    def parameters(self):
        return {key: getattr(self, key) for key in self._parameters_names}
    @parameters.setter
    def parameters(self, values):
        for key, value in values.items():
            setattr(self, key, value)
        self._sync_channels()

    @property
    def constants(self):
        return {key: self._constant_values[key] for key in self._constant_names}
    @constants.setter
    def constants(self, values):
        self._constant_values.update(values)

    @property
    def nmodels(self):
        shape = np.broadcast_shapes(*[np.shape(getattr(self, param, [])) for param in self.parameters])
        if len(shape) > 1:
            raise ValueError('Parameters cannot have more than 1 dimension')
        return shape[0] if len(shape) == 1 else 0

    def _sync_channels(self):
        for pos, channel in enumerate(self.channels):
            values = {}
            values.update({pname: getattr(self, name) for pname, name in self._channel_parameter_names[pos].items()})
            transform = self._transform_parameter_names[pos]
            mux, muy = self.shared
            values[mux] = getattr(self, transform['xx']) * getattr(self, mux) + getattr(self, transform['xy']) * getattr(self, muy) + getattr(self, transform['x0'])
            values[muy] = getattr(self, transform['yx']) * getattr(self, mux) + getattr(self, transform['yy']) * getattr(self, muy) + getattr(self, transform['y0'])
            channel.function.parameters = values

    def prepare_fit_inputs(self, raw_data, args, weights):
        if not isinstance(raw_data, (list, tuple)) or len(raw_data) != len(self.channels):
            raise ValueError('raw_data must contain one stack per channel')
        channel_variables = self._get_channel_variables(args)
        xp = self._get_xp(raw_data, channel_variables, weights)
        raw_arrays, variable_arrays, channels, nmodels = [], [[] for _ in self.channels[0].function.variables], [], None

        for pos, (data, variables) in enumerate(zip(raw_data, channel_variables)):
            data = xp.asarray(data)
            variables = [xp.asarray(var) for var in variables]
            point_shape = np.broadcast_shapes(*[tuple(var.shape) for var in variables])
            if data.shape == point_shape:
                data = data.reshape((1, -1))
            elif data.shape[1:] == point_shape:
                data = data.reshape((data.shape[0], -1))
            else:
                raise ValueError(f'Channel {pos} data shape {data.shape} does not match variables shape {point_shape}')
            if nmodels is None:
                nmodels = data.shape[0]
            elif data.shape[0] != nmodels:
                raise ValueError('All channel data must have the same number of models')
            raw_arrays.append(data)
            npoints = int(np.prod(point_shape))
            channels.append(xp.full(npoints, pos, dtype=xp.int32))
            for vpos, var in enumerate(variables):
                variable_arrays[vpos].append(xp.broadcast_to(var, point_shape).reshape(npoints))

        raw_data = xp.concatenate(raw_arrays, axis=1)
        args = (xp.concatenate(channels), *[xp.concatenate(arrays) for arrays in variable_arrays])
        weights = self._prepare_weights(weights, raw_arrays, channel_variables, xp)
        return raw_data, args, weights

    def _get_channel_variables(self, args):
        if len(args) == 0:
            variables = [channel.variables for channel in self.channels]
        elif len(args) == 1 and len(args[0]) == len(self.channels):
            variables = args[0]
        else:
            raise ValueError('Pass channel variables as [(x0, y0), (x1, y1), ...]')
        for pos, values in enumerate(variables):
            if values is None:
                raise ValueError(f'Missing variables for channel {pos}')
            if len(values) != len(self.channels[0].function.variables):
                raise ValueError(f'Channel {pos} variables do not match function variables')
        return variables

    def _prepare_weights(self, weights, raw_arrays, channel_variables, xp):
        if not isinstance(weights, (list, tuple)):
            return weights
        if len(weights) != len(self.channels):
            raise ValueError('weights must be scalar or one entry per channel')
        arrays = []
        for weight, raw, variables in zip(weights, raw_arrays, channel_variables):
            weight = xp.asarray(weight)
            if weight.size == 1:
                arrays.append(xp.full(raw.shape, weight.reshape(())))
                continue
            point_shape = np.broadcast_shapes(*[tuple(xp.asarray(var).shape) for var in variables])
            if weight.shape == point_shape:
                weight = xp.broadcast_to(weight, point_shape).reshape((1, -1))
            elif weight.shape[1:] == point_shape:
                weight = weight.reshape((weight.shape[0], -1))
            else:
                raise ValueError(f'Weight shape {weight.shape} does not match variables shape {point_shape}')
            arrays.append(xp.broadcast_to(weight, raw.shape))
        return xp.concatenate(arrays, axis=1)

    def _get_xp(self, raw_data, channel_variables, weights):
        try:
            import cupy as cp
        except ImportError:
            cp = None
        if cp is None:
            return np
        arrays = list(raw_data)
        arrays += [var for variables in channel_variables for var in variables]
        if isinstance(weights, (list, tuple)):
            arrays += list(weights)
        else:
            arrays.append(weights)
        return cp if any(isinstance(arr, cp.ndarray) for arr in arrays) else np

    def _build_kernels(self):
        inputs = ', '.join(self.variables + self._parameters_names + self._constant_names)
        self.cpukernel_function = self._build_scalar_kernel(inputs, False, 'function')
        self.gpukernel_function = self._build_scalar_kernel(inputs, True, 'function')
        for pname in self._parameters_names:
            setattr(self, f'cpukernel_d_{pname}', self._build_scalar_kernel(inputs, False, pname))
            setattr(self, f'gpukernel_d_{pname}', self._build_scalar_kernel(inputs, True, pname))
        self.cpu_function = self._build_array_kernel(inputs, False)
        self.gpu_function = self._build_array_kernel(inputs, True)
        self.cpu_jacobian = self._build_jacobian_kernel(inputs, False)
        self.gpu_jacobian = self._build_jacobian_kernel(inputs, True)

    def _build_scalar_kernel(self, inputs, cuda, derivative):
        body = self._scalar_body(derivative)
        string = f'def func({inputs}):\n{body}\n    return 0.0\n'
        glob = self._kernel_globals(cuda)
        loc = {}
        exec(string, glob, loc)
        if cuda:
            return nb.cuda.jit(device=True)(loc['func'])
        return nb.njit(nogil=True)(loc['func'])

    def _build_array_kernel(self, inputs, cuda):
        call = self._indexed_call()
        if cuda:
            string = f'''
def func({inputs}, out, ignore):
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model]:
        out[model, point] = scalar({call})
'''
            glob = {'nb': nb, 'scalar': self.gpukernel_function}
            loc = {}
            exec(string, glob, loc)
            return nb.cuda.jit()(loc['func'])
        string = f'''
def func({inputs}, out, ignore):
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels):
        if ignore[model]:
            continue
        for point in range(npoints):
            out[model, point] = scalar({call})
'''
        glob = {'nb': nb, 'scalar': self.cpukernel_function}
        loc = {}
        exec(string, glob, loc)
        return nb.njit(parallel=True, nogil=True)(loc['func'])

    def _build_jacobian_kernel(self, inputs, cuda):
        call = self._indexed_call()
        if cuda:
            derivatives = '\n'.join([f'''        if bool2fit[{pos}]:\n            jacobian[model, point, count] = d_{key}({call})\n            count += 1''' for pos, key in enumerate(self._parameters_names)])
            string = f'''
def func({inputs}, jacobian, bool2fit, ignore):
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model]:
        count = 0
{derivatives}
'''
            glob = {'nb': nb}
            for key in self._parameters_names:
                glob[f'd_{key}'] = getattr(self, f'gpukernel_d_{key}')
            loc = {}
            exec(string, glob, loc)
            return nb.cuda.jit()(loc['func'])
        derivatives = '\n'.join([f'''            if bool2fit[{pos}]:\n                jacobian[model, point, count] = d_{key}({call})\n                count += 1''' for pos, key in enumerate(self._parameters_names)])
        string = f'''
def func({inputs}, jacobian, bool2fit, ignore):
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels):
        if ignore[model]:
            continue
        for point in range(npoints):
            count = 0
{derivatives}
'''
        glob = {'nb': nb}
        for key in self._parameters_names:
            glob[f'd_{key}'] = getattr(self, f'cpukernel_d_{key}')
        loc = {}
        exec(string, glob, loc)
        return nb.njit(parallel=True, nogil=True)(loc['func'])

    def _indexed_call(self):
        variables = [f'{key}[point]' for key in self.variables]
        parameters = [f'{key}[model]' for key in self._parameters_names]
        return ', '.join(variables + parameters + self._constant_names)

    def _scalar_body(self, derivative):
        body = ''
        for pos, channel in enumerate(self.channels):
            prefix = 'if' if pos == 0 else 'elif'
            body += f'    {prefix} channel == {pos}:\n'
            expr = self._channel_expression(pos, derivative)
            body += f'        return {expr}\n'
        return body

    def _channel_expression(self, pos, derivative):
        channel = self.channels[pos]
        if derivative == 'function':
            return f'f{pos}({self._child_inputs(pos)})'
        if derivative == self.shared[0]:
            transform = self._transform_parameter_names[pos]
            return f'd{pos}_{self.shared[0]}({self._child_inputs(pos)}) * {transform["xx"]} + d{pos}_{self.shared[1]}({self._child_inputs(pos)}) * {transform["yx"]}'
        if derivative == self.shared[1]:
            transform = self._transform_parameter_names[pos]
            return f'd{pos}_{self.shared[0]}({self._child_inputs(pos)}) * {transform["xy"]} + d{pos}_{self.shared[1]}({self._child_inputs(pos)}) * {transform["yy"]}'
        for pname, cname in self._channel_parameter_names[pos].items():
            if derivative == cname:
                return f'd{pos}_{pname}({self._child_inputs(pos)})'
        transform_derivatives = self._transform_derivatives(pos)
        if derivative in transform_derivatives:
            return transform_derivatives[derivative]
        return '0.0'

    def _transform_derivatives(self, pos):
        transform = self._transform_parameter_names[pos]
        mux, muy = self.shared
        child_inputs = self._child_inputs(pos)
        return {
            transform['xx']: f'd{pos}_{mux}({child_inputs}) * {mux}',
            transform['xy']: f'd{pos}_{mux}({child_inputs}) * {muy}',
            transform['x0']: f'd{pos}_{mux}({child_inputs})',
            transform['yx']: f'd{pos}_{muy}({child_inputs}) * {mux}',
            transform['yy']: f'd{pos}_{muy}({child_inputs}) * {muy}',
            transform['y0']: f'd{pos}_{muy}({child_inputs})',
        }

    def _child_inputs(self, pos):
        channel = self.channels[pos]
        values = []
        values += channel.function.variables
        for pname in channel.function.parameters.keys():
            if pname == self.shared[0]:
                values.append(self._local_shared(pos, 0))
            elif pname == self.shared[1]:
                values.append(self._local_shared(pos, 1))
            else:
                values.append(self._channel_parameter_names[pos][pname])
        values += [self._channel_constant_names[pos][key] for key in channel.function.constants.keys()]
        return ', '.join(values)

    def _local_shared(self, pos, axis):
        transform = self._transform_parameter_names[pos]
        mux, muy = self.shared
        if axis == 0:
            return f'({transform["xx"]} * {mux} + {transform["xy"]} * {muy} + {transform["x0"]})'
        return f'({transform["yx"]} * {mux} + {transform["yy"]} * {muy} + {transform["y0"]})'

    def _kernel_globals(self, cuda):
        glob = {'nb': nb}
        for pos, channel in enumerate(self.channels):
            glob[f'f{pos}'] = channel.function.gpukernel_function if cuda else channel.function.cpukernel_function
            for pname in channel.function.parameters.keys():
                glob[f'd{pos}_{pname}'] = getattr(channel.function, f'gpukernel_d_{pname}' if cuda else f'cpukernel_d_{pname}')
        return glob



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)
