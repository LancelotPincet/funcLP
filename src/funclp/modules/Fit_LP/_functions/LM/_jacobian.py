#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import numba as nb



def jacobian(self) :
    processing_unit = 'gpu' if self.cuda else 'cpu'
    jacobian_function = getattr(self, f'{processing_unit}_jacobian', None)
    if jacobian_function is None :
        jacobian_function = gpu_jacobian(self) if self.cuda else cpu_jacobian(self)
        setattr(self, f'{processing_unit}_jacobian', jacobian_function)
    jacobian_function(*self.variables, *self.data, *self.parameters.values(), self.jacobian_data, self.converged)



def cpu_jacobian(self) :
    kernels = {f"d_{parameter}": getattr(self.function, f'cpukernel_d_{parameter}') for parameter in self.parameters2fit}
    inputs = ''
    inputs += ', '.join(self.function.variables) + ', ' if len(self.function.variables) > 0 else ''
    inputs += ', '.join(self.function.data) + ', ' if len(self.function.data) > 0 else ''
    inputs += ', '.join(self.parameters.keys())
    string = f'''
@nb.njit(parallel=True)
def func({inputs}, jacobian, ignore) :
nmodels, npoints, nparams = jacobian.shape
for model in nb.prange(nmodels) :
    if ignore[model] : continue
    for point in range(npoints) :
        for param in range(nparams) :
'''
    for param, parameter in enumerate(self.parameters2fit) :
        string += f'''
            if param == {param} :
                jacobian[model, point, param] = d_{parameter}({inputs})
'''
    glob = {'nb': nb}
    glob.update(kernels)
    loc = {}
    exec(string, glob, loc)
    func = loc['func']
    return func



def gpu_jacobian(self) :
    kernels = {f"d_{parameter}": getattr(self.function, f'cpukernel_d_{parameter}') for parameter in self.parameters2fit}
    inputs = ''
    inputs += ', '.join(self.function.variables) + ', ' if len(self.function.variables) > 0 else ''
    inputs += ', '.join(self.function.data) + ', ' if len(self.function.data) > 0 else ''
    inputs += ', '.join(self.parameters.keys())
    string = f'''
@nb.cuda.jit(parallel=True)
def func({inputs}, jacobian, ignore) :
nmodels, npoints, nparams = jacobian.shape
model, point, param = nb.cuda.grid(3)
if model < nmodels and not ignore[model] and point < npoints and param < nparams :
'''
    for param, parameter in enumerate(self.parameters2fit) :
        string += f'''
    if param == {param} :
        jacobian[model, point, param] = d_{parameter}({inputs})
'''
    glob = {'nb': nb}
    glob.update(kernels)
    loc = {}
    exec(string, glob, loc)
    func = loc['func']
    threads_per_block = 8, 8, 8
    blocks_per_grid = (
        (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
        (self.npoints + threads_per_block[1] - 1) // threads_per_block[1],
        (self.nparameters2fit + threads_per_block[2] - 1) // threads_per_block[2],
        )
    return func[blocks_per_grid, threads_per_block]