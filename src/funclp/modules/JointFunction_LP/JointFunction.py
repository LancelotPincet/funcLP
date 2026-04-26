#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-26
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : JointFunction

"""
This class defines a way to apply joint models between various models.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference, Parameter, JointChannel
import inspect
import numpy as np
import numba as nb



# %% Joint function
class JointFunction(CudaReference):
    '''Composite function used to jointly fit several independent channels.

    Parameters
    ----------
    channels : list
        Functions or JointChannel instances.
    shared_parameters : dict, optional
        Mapping from joint parameter name to per-channel child parameter names.
        Use None for channels that do not share this parameter.
        Example: {'x0': ['mux', None, 'x0'], 'sigma': [None, 'sig', 'sig']}.
    shared_variables : dict, optional
        Mapping from joint variable name to per-channel child variable names.
        Use None for channels that do not use this joint variable.
        Example: {'x': ['x', 'xx'], 'y': ['y', 'yy'], 'z': ['z', None]}.
    kwargs : dict
        Attributes to change after construction.

    Notes
    -----
    Data inputs are intentionally kept channel-local. They are passed to
    prepare_fit_inputs as one data block per channel, because different channels
    may have different data input names and shapes.

    Affine transforms are fixed and applied on variables, not parameters. Since
    affine coefficients are not fitted parameters, no derivatives are generated
    for them.
    '''

    @prop()
    def name(self):
        return self.__class__.__name__

    def __init__(self, channels, shared_parameters=None, shared_variables=None, **kwargs):
        self.channels = [channel if isinstance(channel, JointChannel) else JointChannel(channel) for channel in channels]
        if len(self.channels) < 2:
            raise ValueError('JointFunction needs at least two channels')
        self.shared_parameters = {} if shared_parameters is None else dict(shared_parameters)
        self.shared_variables = {} if shared_variables is None else dict(shared_variables)
        self._setup_channels()
        self._setup_variables_and_data()
        self._setup_parameters()
        self._setup_constants()
        self._setup_fit_ufunc()
        self._build_kernels()
        selfkwargs(self, kwargs)
        self._sync_channels()

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    def _setup_channels(self):
        prefixes = set()
        for pos, channel in enumerate(self.channels):
            channel.prefix = channel.prefix or f'ch{pos}'
            if not channel.prefix.isidentifier():
                raise ValueError(f'{channel.prefix} is not a valid parameter prefix')
            if channel.prefix in prefixes:
                raise ValueError(f'Duplicate channel prefix {channel.prefix}')
            prefixes.add(channel.prefix)

    def _setup_variables_and_data(self):
        self._validate_shared_variables()
        self._channel_variable_exprs = []

        variable_names = ['channel']
        for key in self.shared_variables.keys():
            if not key.isidentifier():
                raise ValueError(f'{key} is not a valid joint variable name')
            _unique_append(variable_names, key)

        for pos, channel in enumerate(self.channels):
            expressions = {}
            for cname in channel.function.variables:
                expressions[cname] = self._mapped_variable_expression(pos, cname)
            self._channel_variable_exprs.append(expressions)

        self.variables = variable_names

        # Data is channel-local. Each data input receives a unique joint name,
        # so the kernels can still have one flat input signature.
        self.data = []
        self._channel_data_names = []
        for channel in self.channels:
            channel_names = {}
            for dname in channel.function.data:
                joint_name = f'{channel.prefix}_{dname}'
                self.data.append(joint_name)
                channel_names[dname] = joint_name
            self._channel_data_names.append(channel_names)

    def _validate_shared_variables(self):
        n = len(self.channels)
        normalized = {}
        for joint_name, mapping in self.shared_variables.items():
            mapping = _as_list(mapping, n, f'shared_variables[{joint_name!r}]')
            normalized[joint_name] = mapping
            for pos, child_name in enumerate(mapping):
                if child_name is None:
                    continue
                if child_name not in self.channels[pos].function.variables:
                    raise ValueError(f'Variable {child_name} from shared variable {joint_name} is missing from channel {pos}')
        self.shared_variables = normalized

        for pos, channel in enumerate(self.channels):
            mapped = {child for values in self.shared_variables.values() for i, child in enumerate(values) if i == pos and child is not None}
            for group in channel.affine.keys():
                for child_name in group:
                    if child_name not in channel.function.variables:
                        raise ValueError(f'Affine variable {child_name} is missing from channel {pos}')
                    if child_name not in mapped:
                        raise ValueError(f'Affine variable {child_name} in channel {pos} must be mapped by shared_variables')

    def _setup_parameters(self):
        self._validate_shared_parameters()
        names = []
        specs = {}

        self._shared_parameter_sources = {}
        for joint_name, mapping in self.shared_parameters.items():
            source = self._shared_source(joint_name, mapping)
            specs[joint_name] = Parameter(
                joint_name,
                source.default,
                estimate=source.estimate,
                bounds=source.bounds,
                fit=source.fit,
            )
            names.append(joint_name)
            self._shared_parameter_sources[joint_name] = source

        self._channel_parameter_names = []
        self._channel_shared_parameter_names = []
        for pos, channel in enumerate(self.channels):
            local_names = {}
            shared_names = {}
            linked_child_params = self._linked_child_parameters(pos)
            for pname, value in channel.function.parameters.items():
                if pname in linked_child_params:
                    shared_names[pname] = linked_child_params[pname]
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
                local_names[pname] = name
            self._channel_parameter_names.append(local_names)
            self._channel_shared_parameter_names.append(shared_names)

        self._parameters_names = names
        self._parameter_specs = specs
        for key in self._parameters_names:
            setattr(self, key, convert(specs[key].default))
            setattr(self, f'{key}_min', -np.float32(np.inf) if specs[key].bounds[0] is None else specs[key].bounds[0])
            setattr(self, f'{key}_max', +np.float32(np.inf) if specs[key].bounds[1] is None else specs[key].bounds[1])
            setattr(self, f'{key}_fit', bool(specs[key].fit))

    def _validate_shared_parameters(self):
        n = len(self.channels)
        normalized = {}
        for joint_name, mapping in self.shared_parameters.items():
            if not joint_name.isidentifier():
                raise ValueError(f'{joint_name} is not a valid shared parameter name')
            mapping = _as_list(mapping, n, f'shared_parameters[{joint_name!r}]')
            normalized[joint_name] = mapping
            found = False
            for pos, child_name in enumerate(mapping):
                if child_name is None:
                    continue
                found = True
                if child_name not in self.channels[pos].function.parameters:
                    raise ValueError(f'Parameter {child_name} from shared parameter {joint_name} is missing from channel {pos}')
            if not found:
                raise ValueError(f'Shared parameter {joint_name} is not linked to any channel')
        self.shared_parameters = normalized

        for pos, channel in enumerate(self.channels):
            seen = {}
            for joint_name, mapping in self.shared_parameters.items():
                child_name = mapping[pos]
                if child_name is None:
                    continue
                if child_name in seen:
                    raise ValueError(f'Channel {pos} parameter {child_name} is linked by both {seen[child_name]} and {joint_name}')
                seen[child_name] = joint_name

    def _shared_source(self, joint_name, mapping):
        for pos, child_name in enumerate(mapping):
            if child_name is None:
                continue
            channel = self.channels[pos]
            spec = channel.function.__class__.function.parameter_specs[child_name]
            return Parameter(
                joint_name,
                channel.function.parameters[child_name],
                estimate=spec.estimate,
                bounds=spec.bounds,
                fit=getattr(channel.function, f'{child_name}_fit'),
            )
        raise ValueError(f'Shared parameter {joint_name} was not found in any channel')

    def _linked_child_parameters(self, pos):
        linked = {}
        for joint_name, mapping in self.shared_parameters.items():
            child_name = mapping[pos]
            if child_name is not None:
                linked[child_name] = joint_name
        return linked

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
        self.fit_ufunc = _FitUfunc(self.variables, self.data, self._parameters_names, self._parameter_specs)

    # ---------------------------------------------------------------------
    # Public properties
    # ---------------------------------------------------------------------
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
            values.update({pname: getattr(self, name) for pname, name in self._channel_shared_parameter_names[pos].items()})
            channel.function.parameters = values

    # ---------------------------------------------------------------------
    # Fit input preparation
    # ---------------------------------------------------------------------
    def prepare_fit_inputs(self, raw_data, args, weights):
        if not isinstance(raw_data, (list, tuple)) or len(raw_data) != len(self.channels):
            raise ValueError('raw_data must contain one stack per channel')

        channel_inputs = self._get_channel_inputs(args)
        xp = self._get_xp(raw_data, channel_inputs, weights)

        raw_arrays = []
        channel_arrays = []
        joint_variable_arrays = [[] for _ in self.variables[1:]]
        joint_data_arrays = [[] for _ in self.data]
        nmodels = None

        for pos, channel in enumerate(self.channels):
            data = xp.asarray(raw_data[pos])
            variables, data_inputs = channel_inputs[pos]
            variables = {key: xp.asarray(value) for key, value in variables.items()}
            data_inputs = {key: xp.asarray(value) for key, value in data_inputs.items()}

            point_shape = self._channel_point_shape(channel, variables)
            data = self._reshape_model_points(data, point_shape, f'Channel {pos} raw_data')
            if nmodels is None:
                nmodels = data.shape[0]
            elif data.shape[0] != nmodels:
                raise ValueError('All channel data must have the same number of models')

            raw_arrays.append(data)
            npoints = int(np.prod(point_shape))
            channel_arrays.append(xp.full(npoints, pos, dtype=xp.int32))

            for joint_pos, joint_name in enumerate(self.variables[1:]):
                child_name = self.shared_variables[joint_name][pos]
                if child_name is None:
                    joint_variable_arrays[joint_pos].append(xp.zeros(npoints, dtype=xp.float32))
                else:
                    arr = xp.broadcast_to(variables[child_name], point_shape).reshape(npoints)
                    joint_variable_arrays[joint_pos].append(arr)

            data_offset = 0
            for cpos, other in enumerate(self.channels):
                for child_dname in other.function.data:
                    if cpos != pos:
                        joint_data_arrays[data_offset].append(xp.zeros((data.shape[0], npoints), dtype=xp.float32))
                    else:
                        arr = self._reshape_data_input(data_inputs[child_dname], data.shape[0], point_shape, f'Channel {pos} data input {child_dname}')
                        joint_data_arrays[data_offset].append(arr)
                    data_offset += 1

        raw_data = xp.concatenate(raw_arrays, axis=1)
        args = (
            xp.concatenate(channel_arrays),
            *[xp.concatenate(arrays) for arrays in joint_variable_arrays],
            *[xp.concatenate(arrays, axis=1) for arrays in joint_data_arrays],
        )
        weights = self._prepare_weights(weights, raw_arrays, channel_inputs, xp)
        return raw_data, args, weights

    def _get_channel_inputs(self, args):
        if len(args) == 0:
            values = [(self._default_channel_variables(channel), {}) for channel in self.channels]
        elif len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == len(self.channels):
            values = [self._normalize_channel_input(pos, item) for pos, item in enumerate(args[0])]
        else:
            raise ValueError('Pass channel inputs as [(variables, data), ...] or [{' + "'x': x, ..." + '}, ...]')

        for pos, (variables, data_inputs) in enumerate(values):
            channel = self.channels[pos]
            for vname in channel.function.variables:
                if vname not in variables:
                    raise ValueError(f'Missing variable {vname} for channel {pos}')
            for dname in channel.function.data:
                if dname not in data_inputs:
                    raise ValueError(f'Missing data input {dname} for channel {pos}')
        return values

    def _normalize_channel_input(self, pos, item):
        channel = self.channels[pos]
        if item is None:
            raise ValueError(f'Missing inputs for channel {pos}')

        if isinstance(item, dict):
            variables = {key: item[key] for key in channel.function.variables if key in item}
            data_inputs = {key: item[key] for key in channel.function.data if key in item}
            return variables, data_inputs

        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], dict):
            variables = dict(item[0])
            data_inputs = dict(item[1] or {})
            return variables, data_inputs

        if isinstance(item, (list, tuple)):
            nvars = len(channel.function.variables)
            ndata = len(channel.function.data)
            if len(item) != nvars + ndata:
                raise ValueError(f'Channel {pos} expects {nvars} variable inputs and {ndata} data inputs')
            variables = dict(zip(channel.function.variables, item[:nvars]))
            data_inputs = dict(zip(channel.function.data, item[nvars:]))
            return variables, data_inputs

        raise ValueError(f'Invalid input format for channel {pos}')

    def _default_channel_variables(self, channel):
        if not channel.variables:
            return None
        reverse = {}
        for joint_name, child_name in channel.variables.items():
            reverse[child_name] = joint_name
        return reverse

    def _channel_point_shape(self, channel, variables):
        shapes = [tuple(np.shape(variables[vname])) for vname in channel.function.variables]
        if len(shapes) == 0:
            return ()
        return np.broadcast_shapes(*shapes)

    def _reshape_model_points(self, data, point_shape, label):
        if data.shape == point_shape:
            return data.reshape((1, -1))
        if data.shape[1:] == point_shape:
            return data.reshape((data.shape[0], -1))
        raise ValueError(f'{label} shape {data.shape} does not match variables shape {point_shape}')

    def _reshape_data_input(self, data, nmodels, point_shape, label):
        if data.shape == point_shape:
            return data.reshape((1, -1))
        if data.shape[1:] == point_shape:
            if data.shape[0] != nmodels:
                raise ValueError(f'{label} has {data.shape[0]} models but expected {nmodels}')
            return data.reshape((data.shape[0], -1))
        raise ValueError(f'{label} shape {data.shape} does not match variables shape {point_shape}')

    def _prepare_weights(self, weights, raw_arrays, channel_inputs, xp):
        if not isinstance(weights, (list, tuple)):
            return weights
        if len(weights) != len(self.channels):
            raise ValueError('weights must be scalar or one entry per channel')
        arrays = []
        for pos, (weight, raw, inputs) in enumerate(zip(weights, raw_arrays, channel_inputs)):
            variables, _ = inputs
            point_shape = self._channel_point_shape(self.channels[pos], variables)
            weight = xp.asarray(weight)
            if weight.size == 1:
                arrays.append(xp.full(raw.shape, weight.reshape(())))
                continue
            if weight.shape == point_shape:
                weight = xp.broadcast_to(weight, point_shape).reshape((1, -1))
            elif weight.shape[1:] == point_shape:
                weight = weight.reshape((weight.shape[0], -1))
            else:
                raise ValueError(f'Weight shape {weight.shape} does not match variables shape {point_shape}')
            arrays.append(xp.broadcast_to(weight, raw.shape))
        return xp.concatenate(arrays, axis=1)

    def _get_xp(self, raw_data, channel_inputs, weights):
        try:
            import cupy as cp
        except ImportError:
            cp = None
        if cp is None:
            return np
        arrays = list(raw_data)
        for variables, data_inputs in channel_inputs:
            arrays += list(variables.values())
            arrays += list(data_inputs.values())
        if isinstance(weights, (list, tuple)):
            arrays += list(weights)
        else:
            arrays.append(weights)
        return cp if any(isinstance(arr, cp.ndarray) for arr in arrays if arr is not None) else np

    # ---------------------------------------------------------------------
    # Kernel construction
    # ---------------------------------------------------------------------
    def _build_kernels(self):
        inputs = ', '.join(self.variables + self.data + self._parameters_names + self._constant_names)
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
        data = [f'{key}[model, point]' for key in self.data]
        parameters = [f'{key}[model]' for key in self._parameters_names]
        return ', '.join(variables + data + parameters + self._constant_names)

    def _scalar_body(self, derivative):
        body = ''
        for pos, channel in enumerate(self.channels):
            prefix = 'if' if pos == 0 else 'elif'
            body += f'    {prefix} channel == {pos}:\n'
            expr = self._channel_expression(pos, derivative)
            body += f'        return {expr}\n'
        return body

    def _channel_expression(self, pos, derivative):
        if derivative == 'function':
            return f'f{pos}({self._child_inputs(pos)})'

        # Shared joint parameters: derivative is the child derivative for each
        # channel where the parameter is linked, otherwise zero.
        if derivative in self.shared_parameters:
            child_name = self.shared_parameters[derivative][pos]
            if child_name is None:
                return '0.0'
            return f'd{pos}_{child_name}({self._child_inputs(pos)})'

        # Channel-local parameters.
        for child_name, joint_name in self._channel_parameter_names[pos].items():
            if derivative == joint_name:
                return f'd{pos}_{child_name}({self._child_inputs(pos)})'

        return '0.0'

    def _child_inputs(self, pos):
        channel = self.channels[pos]
        values = []
        values += [self._channel_variable_exprs[pos][key] for key in channel.function.variables]
        values += [self._channel_data_names[pos][key] for key in channel.function.data]
        for pname in channel.function.parameters.keys():
            if pname in self._channel_shared_parameter_names[pos]:
                values.append(self._channel_shared_parameter_names[pos][pname])
            else:
                values.append(self._channel_parameter_names[pos][pname])
        values += [self._channel_constant_names[pos][key] for key in channel.function.constants.keys()]
        return ', '.join(values)

    def _mapped_variable_expression(self, pos, child_name):
        joint_name = None
        for key, mapping in self.shared_variables.items():
            if mapping[pos] == child_name:
                joint_name = key
                break
        if joint_name is None:
            raise ValueError(f'Channel {pos} variable {child_name} is not mapped by shared_variables')
        return self._affine_variable_expression(pos, child_name, joint_name)

    def _affine_variable_expression(self, pos, child_name, fallback_joint_name):
        channel = self.channels[pos]
        for group, matrix in channel.affine.items():
            if child_name not in group:
                continue
            row = group.index(child_name)
            terms = []
            for col, other_child_name in enumerate(group):
                joint_name = self._joint_variable_for_child(pos, other_child_name)
                coeff = float(matrix[row, col])
                if coeff == 0:
                    continue
                terms.append(f'({coeff!r} * {joint_name})')
            shift = float(matrix[row, len(group)])
            if shift != 0:
                terms.append(f'{shift!r}')
            return '(' + ' + '.join(terms or ['0.0']) + ')'
        return fallback_joint_name

    def _joint_variable_for_child(self, pos, child_name):
        for joint_name, mapping in self.shared_variables.items():
            if mapping[pos] == child_name:
                return joint_name
        raise ValueError(f'Channel {pos} variable {child_name} is not mapped by shared_variables')

    def _kernel_globals(self, cuda):
        glob = {'nb': nb}
        for pos, channel in enumerate(self.channels):
            glob[f'f{pos}'] = channel.function.gpukernel_function if cuda else channel.function.cpukernel_function
            for pname in channel.function.parameters.keys():
                glob[f'd{pos}_{pname}'] = getattr(channel.function, f'gpukernel_d_{pname}' if cuda else f'cpukernel_d_{pname}')
        return glob


def _as_list(value, n, name):
    if value is None:
        return [None] * n
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f'{name} must be a list/tuple of length {n}')
    return list(value)


def _unique_append(values, value):
    if value not in values:
        values.append(value)



class _FitUfunc:
    def __init__(self, variables, data, parameters, parameter_specs):
        self.variables = variables
        self.data = data
        self.parameters = parameters
        self.parameter_specs = parameter_specs
        signature_parameters = [inspect.Parameter(key, inspect.Parameter.POSITIONAL_ONLY) for key in variables]
        signature_parameters += [inspect.Parameter(key, inspect.Parameter.POSITIONAL_ONLY) for key in data]
        signature_parameters += [inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=parameter_specs[key].default) for key in parameters]
        self.signature = inspect.Signature(signature_parameters)



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
