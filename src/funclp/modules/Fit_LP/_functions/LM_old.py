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
    damping_init = 1. #lm damping
    damping_increase, damping_decrease = 7.0, 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    max_retries = 10 # Maximum of step retries for a given hessian

    # Functions

    def fit_init(self) :
        self.damping_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=self.damping_init)
        self.hessian_cache = self.xp.empty_like(self.hessian_data)
        
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

            # --- TEMP DIAGNOSTICS ---
            if retry == 0:
                from corelp import Path
                path = Path(r'C:\Users\LPINCET\Desktop\modpro_data\acq_tub_600mW_200ms_m100Hz') / 'debug.txt'
                debug_file = getattr(self, 'debug', None)
                if debug_file is None :
                    with open(path, 'w') as f:
                        f.write('')
                    self.debug = True
                predicted = 0.0
                for p in range(self.nparameters2fit):
                    predicted += float(self.parameter_steps[0, p]) * (
                        float(self.damping_data[0]) * float(self.parameter_steps[0, p])
                        - float(self.gradient_data[0, p])
                    )
                predicted *= 0.5
                actual = float(self.chi2_cache[0]) - float(self.chi2_data[0])
                lines = [
                    f"\n--- First retry, iteration diagnostics ---",
                    f"damping:    {self.damping_data[:5]}",
                    f"hessian diag[0]: {self.hessian_cache[0].diagonal()}",
                    f"gradient[0]: {self.gradient_data[0]}",
                    f"steps[0]:   {self.parameter_steps[0]}",
                    f"chi2_cache: {self.chi2_cache[:5]}",
                    f"chi2_new:   {self.chi2_data[:5]}",
                    f"predicted:  {predicted:.6e}",
                    f"actual:     {actual:.6e}",
                    f"rho:        {actual/predicted if abs(predicted)>1e-12 else 'undefined'}",
                    f"failed:     {self.failed[:5]}",
                    f"improved:   {self.improved[:5]}",
                ]
                with open(path, 'a') as f:
                    f.write('\n'.join(lines) + '\n')
                # --- END DIAGNOSTICS ---

            # Checking if improved
            self.improving()
            if self.improved.all() :
                break



    # --- Damping ---

    def damping(self) :
        damping_function = self.gpu_damping if self.cuda else self.cpu_damping
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
                        if param == paramT :
                            hessian[model, param, paramT] += damping[model] * hessian[model, param, paramT]
        return func

    @prop(cache=True)
    def gpu_damping(self) :
        @nb.cuda.jit()
        def func(hessian, hessian_cache, damping, ignore) :
            nmodels, nparams, _ = hessian.shape
            model, param, paramT = nb.cuda.grid(3)
            if model < nmodels and not ignore[model] and param < nparams and paramT < nparams :
                hessian[model, param, paramT] = hessian_cache[model, param, paramT] 
                if param == paramT :
                    hessian[model, param, paramT] += damping[model] * hessian[model, param, paramT]
        threads_per_block = 8, 8, 8
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            (self.nparameters2fit + threads_per_block[2] - 1) // threads_per_block[2],
            )
        return func[blocks_per_grid, threads_per_block]



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
                # Max step test TODO in GPU
                if not ignore[model]:
                    norm = 0.0
                    for param in range(nparams):
                        norm += steps[model, param] ** 2
                    norm = math.sqrt(norm)
                    if norm > 10.0:
                        scale = 10.0 / norm
                        for param in range(nparams):
                            steps[model, param] *= scale
        return func

    @prop(cache=True)
    def gpu_solve(self) :
        @nb.cuda.jit()
        def func(hessian, gradient, steps, ignore) :
            model = nb.cuda.blockIdx.x
            tid = nb.cuda.threadIdx.x
            nparams = hessian.shape[1]
            if ignore[model]:
                return
            # --- Cholesky factorization ---
            for i in range(nparams):
                # Compute diagonal element
                if tid == i:
                    s = hessian[model, i, i]
                    for k in range(i):
                        s -= hessian[model, i, k] * hessian[model, i, k]
                    if s <= 0.0:
                        ignore[model] = True
                    else:
                        hessian[model, i, i] = math.sqrt(s)
                nb.cuda.syncthreads()
                if ignore[model]:
                    return
                # Compute column below diagonal
                if tid > i and tid < nparams:
                    s = hessian[model, tid, i]
                    for k in range(i):
                        s -= hessian[model, tid, k] * hessian[model, i, k]
                    hessian[model, tid, i] = s / hessian[model, i, i]
                nb.cuda.syncthreads()
            # --- Forward substitution: L y = -g ---
            for i in range(nparams):
                if tid == i:
                    s = -gradient[model, i]
                    for k in range(i):
                        s -= hessian[model, i, k] * steps[model, k]
                    steps[model, i] = s / hessian[model, i, i]
                nb.cuda.syncthreads()
            # --- Back substitution: Lᵀ x = y ---
            for i in range(nparams - 1, -1, -1):
                if tid == i:
                    s = steps[model, i]
                    for k in range(i + 1, nparams):
                        s -= hessian[model, k, i] * steps[model, k]
                    steps[model, i] = s / hessian[model, i, i]
                nb.cuda.syncthreads()
        threads_per_block = 64
        blocks_per_grid = self.nmodels
        return func[blocks_per_grid, threads_per_block]



    # --- Parameter change ---

    def parameter_change(self) :
        change_function = self.gpu_paramchange if self.cuda else self.cpu_paramchange
        change_function(self.parameters.T, self.parameter_steps, self.failed)

    @prop(cache=True)
    def cpu_paramchange(self) :
        @nb.njit(parallel=True)
        def func(parameters, parameter_steps, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    parameters[model, param] += parameter_steps[model, param]
        return func

    @prop(cache=True)
    def gpu_paramchange(self) :
        @nb.cuda.jit()
        def func(parameters, parameter_steps, ignore) :
            nmodels, nparams = parameters.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
                parameters[model, param] += parameter_steps[model, param]
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
        improving_function(self.chi2_cache, self.chi2_data, self.parameters.T, self.parameter_steps, self.gradient_data, self.damping_data, self.damping_increase, self.damping_decrease, self.damping_max, self.damping_min, self.converged, self.improved, self.failed)

    @prop(cache=True)
    def cpu_improve(self) :
        @nb.njit(parallel=True)
        def func(old_chi2, new_chi2, parameters, parameter_steps, gradient, damping, damping_increase, damping_decrease, damping_max, damping_min, converged, improved, failed) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if improved[model] : continue
                if failed[model] :  # Cholesky failed — just increase damping, no revert needed
                    damping[model] = min(damping[model] * damping_increase, damping_max)
                    if damping[model] >= damping_max :
                        converged[model] = -1
                        improved[model] = True
                    continue
                # Check step more than zero
                step_norm = 0.0
                for param in range(nparams):
                    step_norm += parameter_steps[model, param] ** 2
                if step_norm < 1e-28:  # Step is numerically zero
                    converged[model] = -1
                    improved[model] = True
                    continue
                # Compute predicted reduction: (1/2) * p^T * (λ*p - g)
                predicted = 0.0
                for param in range(nparams):
                    predicted += parameter_steps[model, param] * (
                        damping[model] * parameter_steps[model, param] - gradient[model, param]
                    )
                predicted *= 0.5
                actual_reduction = old_chi2[model] - new_chi2[model]
                rho = actual_reduction / predicted if predicted > 1e-12 else -1.0
                if rho > 0.0 :
                    improved[model] = True
                    if rho > 0.75 :
                        damping[model] = max(damping[model] * damping_decrease, damping_min)
                    elif rho < 0.25 :
                        damping[model] = min(damping[model] * damping_increase, damping_max)
                else :
                    for param in range(nparams) :
                        parameters[model, param] -= parameter_steps[model, param]
                    damping[model] = min(damping[model] * damping_increase, damping_max)
                    if damping[model] >= damping_max :
                        converged[model] = -1
                        improved[model] = True
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

