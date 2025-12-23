#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Fit
from corelp import prop
import numpy as np
import numba as nb
from numba import cuda



# %% Function

class LM(Fit) :

    # Attributs
    ftol = 1e-6 # Loop stops when chi2 change is lower than ftol.
    xtol = 0 # Loop stops when parameter step is lower than xtol.
    gtol = 0 # Loop stops when gradient maximum is lower than gtol.
    damping_init = 1e-2 #lm damping
    damping_increase = 7.0 # Factor to apply when damping
    damping_decrease = 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    retries_max = 10 # Maximum of retries for a given hessian

    def fit(self) :

        # Allocating data

        self.mode_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)
        self.jacobian_data = self.xp.empty(shape=(self.nmodels, self.npoints, self.nparameters2fit), dtype=self.dtype)

        self.chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype)
        self.new_chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype)
        self.deviance_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)

        self.gradient_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, 1), dtype=self.dtype)
        self.loss_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)

        self.hessian_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype)
        self.fisher_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)

        self.damping_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=self.damping_init)

        # Iterations
        for _ in range(self.max_iterations) :

            # Calculate current model data and jacobian
            self.function(*self.variables, *self.data, **self.parameters, out=self.model_data, ignore=self.converged)
            self.jacobian(self.jacobian_data)

            # Evaluate Solving arrays
            self.chi2(self.chi2_data)
            self.gradient(self.gradient_data)
            self.hessian(self.hessian_data)
            
            # Solving


    # --- Jacobian ---



    def jacobian(self, out) :
        jacobian_function = self.gpu_jacobian if self.cuda else self.cpu_jacobian
        jacobian_function(*self.variables, *self.data, *self.parameters.values(), out, self.converged)

    @prop(cache=True)
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

    @prop(cache=True)
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



    # --- Chi² ---



    def chi2(self, out) :
        self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.converged)
        deviance2chi2 = self.gpu_deviance2chi2 if self.cuda else self.cpu_deviance2chi2
        deviance2chi2(self.deviance_data, out, self.converged)

    @prop(cache=True)
    def cpu_deviance2chi2(self) :
        @nb.njit(parallel=True)
        def func(deviance, chi2, converged) :
            nmodels, npoints = deviance.shape
            for model in nb.prange(nmodels) :
                if converged[model] : continue
                s = 0.0
                for point in range(npoints) :
                    s += deviance[model, point]
                chi2[model] = s
        return func

    @prop(cache=True)
    def gpu_deviance2chi2(self) :
        @nb.cuda.jit()
        def func(deviance, chi2, converged) :
            nmodels, npoints = deviance.shape
            model = cuda.grid(1)  # 1D grid of threads
            if model < nmodels and not converged[model] :
                s = 0.0
                for point in range(npoints):
                    s += deviance[model, point]
                chi2[model] = s
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]



    # --- Gradient ---



    def gradient(self, out) :
        self.estimator.loss(self.raw_data, self.model_data, weights=self.weights, out=self.loss_data, ignore=self.converged)
        loss2gradient = self.gpu_loss2gradient if self.cuda else self.cpu_loss2gradient
        loss2gradient(self.jacobian_data, self.loss_data, out, self.converged)

    @prop(cache=True)
    def cpu_loss2gradient(self) :
        @nb.njit(parallel=True)
        def func(jacobian, loss, gradient, converged) :
            nmodels, npoints, nparams = jacobian.shape
            for model in nb.prange(nmodels) :
                if converged[model] : continue
                for param in range(nparams) :
                    s = 0.0
                    for point in range(npoints) :
                        s += jacobian[model, point, param] * loss[model, point]
                    gradient[model, param, 0] = s
        return func

    @prop(cache=True)
    def gpu_loss2gradient(self) :
        @nb.cuda.jit()
        def func(jacobian, loss, gradient, converged) :
            nmodels, npoints, nparams = jacobian.shape
            model, param = cuda.grid(2)
            if model < nmodels and not converged[model] and param < nparams :
                s = 0.0
                for point in range(npoints):
                    s += jacobian[model, point, param] * loss[model, point]
                gradient[model, param, 0] = s
        threads_per_block = 16, 16
        blocks_per_grid = (
                (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
                (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
                )
        return func[blocks_per_grid, threads_per_block]



    # --- Hessian ---



    def hessian(self, out) :
        self.estimator.fisher(self.raw_data, self.model_data, weights=self.weights, out=self.fisher_data, ignore=self.converged)
        fisher2ghessian = self.gpu_fisher2ghessian if self.cuda else self.cpu_fisher2ghessian
        fisher2ghessian(self.jacobian_data, self.fisher_data, out, self.converged)

    @prop(cache=True)
    def cpu_loss2gradient(self) :
        @nb.njit(parallel=True)
        def func(jacobian, fisher, hessian, converged) :
            nmodels, npoints, nparams = jacobian.shape
            for model in nb.prange(nmodels) :
                if converged[model] : continue
                for param in range(nparams) :
                    for paramT in range(nparams) :
                        s = 0.0
                        for point in range(npoints) :
                            s += jacobian[model, point, param] * jacobian[model, point, paramT] * fisher[model, point]
                        hessian[model, param, paramT] = s
        return func

    @prop(cache=True)
    def gpu_loss2gradient(self) :
        @nb.cuda.jit()
        def func(jacobian, fisher, hessian, converged) :
            nmodels, npoints, nparams = jacobian.shape
            model, param, paramT = cuda.grid(3)
            if model < nmodels and not converged[model] and param < nparams and paramT < nparams :
                s = 0.0
                for point in range(npoints):
                    s += jacobian[model, point, param] * jacobian[model, point, paramT] * fisher[model, point]
                hessian[model, param, paramT] = s
        threads_per_block = 8, 8, 8
        blocks_per_grid = (
                (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
                (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
                (self.nparameters2fit + threads_per_block[2] - 1) // threads_per_block[2],
                )
        return func[blocks_per_grid, threads_per_block]



    # --- Damping ---



    def damping(self, out) :
        damping_function = self.gpu_damping if self.cuda else self.cpu_damping
        damping_function(self.damping_data, out, self.converged)

    @prop(cache=True)
    def cpu_damping(self) :
        @nb.njit(parallel=True)
        def func(damping, hessian, converged) :
            nmodels, nparams, nparamsT = hessian.shape
            for model in nb.prange(nmodels) :
                if converged[model] : continue
                for param in range(nparams) :
                    hessian[model, param, param] += damping[model]
        return func

    @prop(cache=True)
    def gpu_damping(self) :
        @nb.cuda.jit()
        def func(damping, hessian, converged) :
            nmodels, nparams, nparamsT = hessian.shape
            model, param = cuda.grid(2)
            if model < nmodels and not converged[model] and param < nparams :
                hessian[model, param, param] += damping[model]
        threads_per_block = 16, 8
        blocks_per_grid = (
                (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
                (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
                )
        return func[blocks_per_grid, threads_per_block]



