#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Fit



# %% Function

class LM(Fit) :

    # Attributs
    damping_init = 1e-2 #lm damping
    damping_increase = 7.0 # Factor to apply when damping
    damping_decrease = 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    max_retries = 10 # Maximum of step retries for a given hessian

    # Functions

    def fit_init(self) :
        self.improved = self.xp.zeros_like(self.converge)
        self.damping_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=self.damping_init)
        self.hessian_diag = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype)
        
    def fit_optimize(self)
        self.subconverged[:] = self.converged
        self.hessian_Hdiag()

        # Looping several time on a given hessian with different dampings
        for _ in range(self.max_retries) :

            # Apply damping : H = Hessian + damping * Identity
            self.damping()

            # Solve function
            self.parameter_steps = self.xp.linalg.solve(self.hessian_data, -self.gradient_data) # TODO

            # Apply steps
            self.parameter_change()

            # evaluate chi2 after step
            self.model()
            self.chi2()

            # Improved
            self.improved[self.improved] = self.chi2_data  self.chi2_cache






    # --- Hessian diagonal ---

    def hessian_Hdiag(self) :
        Hdiag_function = self.gpu_Hdiag if self.cuda else self.cpu_Hdiag
        Hdiag_function(self.hessian, self.hessian_diag, self.converged)

    @prop(cache=True)
    def cpu_Hdiag(self) :
        @nb.njit(parallel=True)
        def func(hessian, hessian_diag, ignore) :
            nmodels, nparams = hessian_diag.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    hessian_diag[model, param] = hessian[model, param, param]
        return func

    @prop(cache=True)
    def gpu_Hdiag(self) :
        @nb.cuda.jit()
        def func(hessian, hessian_diag, ignore) :
            nmodels, nparams = hessian_diag.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                hessian_diag[model, param] = hessian[model, param, param]
        threads_per_block = 16, 16
        blocks_per_grid = (
                (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
                (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
                )
        return func[blocks_per_grid, threads_per_block]



    # --- Damping ---

    def damping(self) :
        damping_function = self.gpu_damping if self.cuda else self.cpu_damping
        damping_function(self.hessian, self.hessian_diag, self.damping_data, self.improved)

    @prop(cache=True)
    def cpu_damping(self) :
        @nb.njit(parallel=True)
        def func(hessian, hessian_diag, damping, ignore) :
            nmodels, nparams = hessian_diag.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    hessian[model, param, param] = hessian_diag[model, param] + damping[model]
        return func

    @prop(cache=True)
    def gpu_damping(self) :
        @nb.cuda.jit()
        def func(hessian, hessian_diag, damping, ignore) :
            nmodels, nparams = hessian_diag.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                hessian[model, param, param] = hessian_diag[model, param] + damping[model]
        threads_per_block = 16, 16
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            )
        return func[blocks_per_grid, threads_per_block]



    # --- Parameter change ---

    def parameter_change(self) :
        change_function = self.gpu_paramchange if self.cuda else self.cpu_paramchange
        change_function(self.parameters, self.parameter_steps, self.ignore)

    @prop(cache=True)
    def cpu_paramchange(self) :
        @nb.njit(parallel=True)
        def func(parameters, parameter_steps, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    parameters[model, param] += parameter_steps[model, param, 0]
        return func

    @prop(cache=True)
    def gpu_paramchange(self) :
        @nb.cuda.jit()
        def func(, ignore) :
            nmodels, nparams = parameters.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                parameters[model, param] += parameter_steps[model, param, 0]
        threads_per_block = 16, 16
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            )
        return func[blocks_per_grid, threads_per_block]



    # --- Improving --- #TODO

    def improving(self) :
        improving = self.gpu_paramchange if self.cuda else self.cpu_paramchange
        change_function(self.parameters, self.parameter_steps, self.ignore)

    @prop(cache=True)
    def cpu_paramchange(self) :
        @nb.njit(parallel=True)
        def func(parameters, parameter_steps, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    parameters[model, param] += parameter_steps[model, param, 0]
        return func

    @prop(cache=True)
    def gpu_paramchange(self) :
        @nb.cuda.jit()
        def func(, ignore) :
            nmodels, nparams = parameters.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                parameters[model, param] += parameter_steps[model, param, 0]
        threads_per_block = 16, 16
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            )
        return func[blocks_per_grid, threads_per_block]

