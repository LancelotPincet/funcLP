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
    damping_increase, damping_decrease = 7.0, 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    max_retries = 10 # Maximum of step retries for a given hessian

    # Functions

    def fit_init(self) :
        self.damping_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=self.damping_init)
        self.damping_cache = self.xp.empty_like(self.damping_data)
        self.hessian_diag = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype)
        
    def fit_optimize(self)

        # Reset cache
        self.damping_cache[:] = self.damping_data
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
            self.evaluate_chi2()

            # Checking if improved
            self.improving()
            if self.improved.all() :
                break

        # Update damping
        self.update_damping()




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
        change_function(self.parameters, self.parameter_steps, self.improved)

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
        def func(parameters, parameter_steps, ignore) :
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



    # --- Evaluate chi2 ---

    def evaluate_chi2(self) :
        self.function(*self.variables, *self.data, **self.parameters, out=self.model_data, ignore=self.improved)
        self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.improved)
        deviance2chi2_function = self.gpu_deviance2chi2 if self.cuda else self.cpu_deviance2chi2
        deviance2chi2_function(self.deviance_data, self.chi2_data, self.improved)



    # --- Improving ---

    def improving(self) :
        improving_function = self.gpu_improve if self.cuda else self.cpu_improve
        improving_function(self.chi2_cache, self.chi2_data, self.parameters, self.parameter_steps, self.damping_data, self.damping_increase, self.damping_max, self.self.improved)

    @prop(cache=True)
    def cpu_improve(self) :
        @nb.njit(parallel=True)
        def func(old_chi2, new_chi2, parameters, parameter_steps, damping, damping_increase, damping_max, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                if old_chi2 < new_chi2 : #Better
                    ignore[model] = True
                else : #Worse
                    for param in range(nparams) :
                        parameters[model, param] -= parameter_steps[model, param, 0] # Come back to before
                    damping[model] *= damping_increase
                    if damping[model] > damping_max :
                        damping[model] = damping_max
        return func

    @prop(cache=True)
    def gpu_improve(self) :
        @nb.cuda.jit()
        def func(old_chi2, new_chi2, parameters, parameter_steps, damping, damping_increase, damping_max, ignore) :
            nmodels, nparams = parameters.shape
            model = nb.cuda.grid(1)
            if model < nmodels and not ignore[model] :
                if old_chi2 < new_chi2 : #Better
                    ignore[model] = True
                else : #Worse
                    for param in range(nparams) :
                        parameters[model, param] -= parameter_steps[model, param, 0] # Come back to before
                    damping[model] *= damping_increase
                    if damping[model] > damping_max :
                        damping[model] = damping_max
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]



    # --- Update damping ---

    def update_damping(self) :
        updatedamp_function = self.gpu_updatedamp if self.cuda else self.cpu_updatedamp
        updatedamp_function(self.damping, self.damping_cache, self.damping_decrease, self.damping_increase, self.damping_min, self.damping_max, self.self.improved, self.converged)

    @prop(cache=True)
    def cpu_updatedamp(self) :
        @nb.njit(parallel=True)
        def func(damping, damping_cache, damping_decrease, damping_increase, damping_min, damping_max, improved, ignore) :
            nmodels, = improved.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                if improved[model] :
                    damping[model] *= damping_decrease
                    if damping[model] < damping_min :
                        damping[model] = damping_min
                else :
                    damping[model] *= damping_increase
                    if damping[model] > damping_max :
                        damping[model] = damping_max


        return func

    @prop(cache=True)
    def gpu_updatedamp(self) :
        @nb.cuda.jit()
        def func(old_chi2, new_chi2, parameters, parameter_steps, damping, damping_increase, damping_max, ignore) :

        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]

