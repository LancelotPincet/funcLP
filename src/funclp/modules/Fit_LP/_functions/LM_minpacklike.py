#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Fit
import math
from corelp import prop
import numba as nb



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
        self.hessian_cache = self.xp.empty_like(self.hessian_data)
        self.rho_data = self.xp.zeros(shape=self.nmodels, dtype=self.dtype) # TODO remove just for debug
        self.nu_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=2.0)       
    
    def fit_optimize(self) :

        # Reset cache
        self.hessian_cache[:] = self.hessian_data

        # Looping several time on a given hessian with different dampings
        for retry in range(self.max_retries) :
            self.failed[:] = self.improved

            # Apply damping : H = Hessian + damping * Identity
            self.damping()

            # Solve function
            self.solve()

            # Apply steps
            self.parameter_change()

            # evaluate chi2 after step
            self.evaluate_chi2()

            # Checking if improved
            self.improving()
            
            # --- TEMP DIAGNOSTICS ---
            if retry == 0:
                from corelp import Path
                path = Path(r'C:\Users\LPINCET\Desktop\modpro_data\acq_tub_600mW_200ms_m100Hz') / 'debug.txt'
                debug_file = getattr(self, 'debug', None)
                if debug_file is None :
                    with open(path, 'w') as f:
                        f.write('')
                    self.debug = True
                lines = [
                    f"\n--- First retry, iteration diagnostics ---",
                    f"damping:    {self.damping_data[:5]}",
                    f"hessian diag[0]: {self.hessian_cache[0].diagonal()}",
                    f"gradient[0]: {self.gradient_data[0]}",
                    f"steps[0]:   {self.parameter_steps[0]}",
                    f"chi2_cache: {self.chi2_cache[:5]}",
                    f"chi2_new:   {self.chi2_data[:5]}",
                    f"rho:     {self.rho_data[:5]}",
                    f"failed:     {self.failed[:5]}",
                    f"improved:   {self.improved[:5]}",
                ]
                with open(path, 'a') as f:
                    f.write('\n'.join(lines) + '\n')
                # --- END DIAGNOSTICS ---

            if self.improved.all() :
                break



    # --- Damping ---

    def damping(self) :
        damping_function = self.cpu_damping
        damping_function(self.hessian_data, self.hessian_cache, self.damping_data, self.improved)

    @prop(cache=True)
    def cpu_damping(self) :
        @nb.njit(parallel=True)
        def func(hessian, hessian_cache, damping, ignore) :
            nmodels, nparams, _ = hessian.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    for paramT in range(nparams) :
                        hessian[model, param, paramT] = hessian_cache[model, param, paramT]
                    hessian[model, param, param] += damping[model]
        return func



    # --- Cholesky Solve ---

    def solve(self) :
        solve_function = self.gpu_solve if self.cuda else self.cpu_solve
        solve_function(self.hessian_data, self.gradient_data, self.parameter_steps, self.failed)

    @prop(cache=True)
    def cpu_solve(self) :
        @nb.njit(parallel=True)
        def func(hessian, gradient, steps, ignore) :
            nmodels, nparams, _ = hessian.shape
            for model in nb.prange(nmodels):
                if ignore[model]: continue
                # --- Cholesky factorization: H = L Lᵀ (in place) ---
                for i in range(nparams):
                    for j in range(i + 1):
                        s = hessian[model, i, j]
                        for k in range(j):
                            s -= hessian[model, i, k] * hessian[model, j, k]
                        if i == j:
                            if s <= 0.0:
                                ignore[model] = True
                                break
                            hessian[model, i, j] = math.sqrt(s)
                        else:
                            hessian[model, i, j] = s / hessian[model, j, j]
                    if ignore[model]:
                        break
                if ignore[model]:
                    continue
                # --- Forward substitution: L y = -g ---
                for i in range(nparams):
                    s = -gradient[model, i]
                    for k in range(i):
                        s -= hessian[model, i, k] * steps[model, k]
                    steps[model, i] = s / hessian[model, i, i]
                # --- Back substitution: Lᵀ x = y ---
                for i in range(nparams - 1, -1, -1):
                    s = steps[model, i]
                    for k in range(i + 1, nparams):
                        s -= hessian[model, k, i] * steps[model, k]
                    steps[model, i] = s / hessian[model, i, i]
        return func


    # --- Parameter change ---

    def parameter_change(self) :
        change_function = self.gpu_paramchange if self.cuda else self.cpu_paramchange
        change_function(self.parameters.T, self.parameter_steps, self.bounds_min, self.bounds_max, self.failed)

    @prop(cache=True)
    def cpu_paramchange(self) :
        @nb.njit(parallel=True)
        def func(parameters, parameter_steps, bounds_min, bounds_max, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    val = parameters[model, param] + parameter_steps[model, param]
                    if val < bounds_min[param]:
                        val = bounds_min[param]
                    elif val > bounds_max[param]:
                        val = bounds_max[param]
                    parameters[model, param] = val
        return func

    @prop(cache=True)
    def gpu_paramchange(self) :
        @nb.cuda.jit()
        def func(parameters, parameter_steps, bounds_min, bounds_max, ignore) :
            nmodels, nparams = parameters.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                val = parameters[model, param] + parameter_steps[model, param]
                if val < bounds_min[param]:
                    val = bounds_min[param]
                elif val > bounds_max[param]:
                    val = bounds_max[param]
                parameters[model, param] = val
        threads_per_block = 16, 16
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            )
        return func[blocks_per_grid, threads_per_block]



    # --- Evaluate chi2 ---

    def evaluate_chi2(self) :
        self.jitted(*self.variables, *self.data, *self.parameters, *self.constants, self.model_data, self.failed)
        self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.failed)
        deviance2chi2_function = self.gpu_deviance2chi2 if self.cuda else self.cpu_deviance2chi2
        deviance2chi2_function(self.deviance_data, self.chi2_data, self.failed)



    # --- Improving ---

    def improving(self) :
        improving_function = self.gpu_improve if self.cuda else self.cpu_improve
        improving_function(self.rho_data, self.nu_data, self.chi2_cache, self.chi2_data, self.parameters.T, self.parameter_steps, self.gradient_data, self.damping_data, self.damping_used, self.diag_scale, self.damping_increase, self.damping_decrease, self.damping_max, self.damping_min, self.converged, self.improved, self.failed)

    @prop(cache=True)
    def cpu_improve(self):
        @nb.njit(parallel=True)
        def func(rho_data, nu_data, old_chi2, new_chi2, parameters, parameter_steps,
                gradient, damping, damping_used, scale,
                damping_increase, damping_decrease, damping_max, damping_min,
                converged, improved, failed):
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels):
                if improved[model]: continue
                pred = 0.0
                for j in range(nparams):
                    sj = parameter_steps[model, j]
                    d2 = scale[model, j] * scale[model, j]
                    pred += sj * (-gradient[model, j] - damping_used[model] * d2 * sj)
                pred *= 0.5
                ared = old_chi2[model] - new_chi2[model]
                rho = ared / pred if abs(pred) > 1e-12 else 0.0
                rho = min(max(rho, -1e6), 1e6)
                rho_data[model] = rho
                if not failed[model] and rho > 0.0:
                    improved[model] = True
                    tmp = 1.0 - (2.0 * rho - 1.0) ** 3
                    if tmp < 1.0 / 3.0: tmp = 1.0 / 3.0
                    elif tmp > 10.0:    tmp = 10.0
                    damping[model] *= tmp
                    nu_data[model] = 2.0
                    if damping[model] < damping_min: damping[model] = damping_min
                else:
                    if not failed[model]:
                        for p in range(nparams):
                            parameters[model, p] -= parameter_steps[model, p]
                    damping[model] *= nu_data[model]
                    nu_data[model] *= 2.0
                    if damping[model] > damping_max: damping[model] = damping_max
        return func

    @prop(cache=True)
    def gpu_improve(self) :
        @nb.cuda.jit()
        def func(old_chi2, new_chi2, parameters, parameter_steps, gradient, damping, damping_increase, damping_decrease, damping_max, damping_min, converged, improved, failed) :
            nmodels, nparams = parameters.shape
            model = nb.cuda.grid(1)
            if model < nmodels and not improved[model] :
                if not failed[model] and old_chi2[model] > new_chi2[model] : #Better
                    improved[model] = True
                    damping[model] *= damping_decrease
                    if damping[model] < damping_min :
                        damping[model] = damping_min
                else : #Worse
                    if not failed[model] :
                        for param in range(nparams) :
                            parameters[model, param] -= parameter_steps[model, param]
                    if damping[model] == damping_max:
                        converged[model] = -1
                        improved[model] = True # Did not really improve but we give up
                    else :
                        damping[model] *= damping_increase
                        if damping[model] > damping_max :
                            damping[model] = damping_max
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]
