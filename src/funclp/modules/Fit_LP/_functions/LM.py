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
    damping_tau = 1e-3 # damping initialization
    damping_min, damping_max = 1e-6, 1e6 # damping limit values
    max_retries = 10 # Maximum of step retries for a given hessian

    # Functions

    def fit_init(self) :        
        self.rho_data = self.xp.zeros(shape=self.nmodels, dtype=self.dtype) # TODO remove just for debug
        self.damping_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype) # Damping data
        self.nu_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=2.0) # Daing factor value
        self.improved = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer loop improved --> Stop inner loop
        self.ignore = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer failed or improved --> continue to next inner loop
        self.damping_initialized = False

    def fit_optimize(self) :

        # Status
        self.xp.not_equal(self.converged, 0, out=self.improved) # Global converged are considered already improved in the inner loop

        # Looping several time on a given hessian with different dampings
        for retry in range(self.max_retries) :
            self.ignore[:] = self.improved # If already improved, we ignore from start

            # Apply damping : H = Hessian + damping * Identity
            self.damping(self.hessian_data, self.hessian_cache, self.damping_data, self.damping_min, self.damping_max, self.damping_tau, self.damping_initialized, self.improved)
            if not self.damping_initialized : self.damping_initialized = True

            # Solve function
            self.solve(self.hessian_data, self.gradient_data, self.parameters_steps, self.ignore)

            # Apply steps
            self.paramchange(self.parameters.T, self.parameters_indices, self.parameters_steps, self.bounds_min, self.bounds_max, self.ignore)

            # evaluate chi2 after step
            self.jitted(*self.variables, *self.data, *self.parameters, *self.constants, self.model_data, self.ignore)
            self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.ignore)
            self.deviance2chi2(self.deviance_data, self.chi2_data, self.ignore)

            # Checking if improved
            self.improving(self.rho_data, self.chi2_cache, self.chi2_data, self.parameters.T, self.parameters_indices, self.parameters_steps, self.gradient_data, self.hessian_cache, self.damping_data, self.nu_data, self.damping_max, self.damping_min, self.converged, self.improved, self.ignore)
            if self.improved.all() :
                break
        self.converged[~self.improved] = -3



    # --- Damping ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_damping(hessian, hessian_cache, damping, damping_min, damping_max, tau, initialized, improved) :
        nmodels, nparams, _ = hessian.shape
        for model in nb.prange(nmodels) :
            if improved[model] : continue
            # Initialization
            if not initialized :
                m = 0.0
                for param in range(nparams) :
                    v = max(1e-12, abs(hessian_cache[model, param, param]))
                    if v > m : m = v
                damping[model] = min(max(tau * m, damping_min), damping_max)
            # Damping
            for param in range(nparams) :
                for paramT in range(nparams) :
                    hessian[model, param, paramT] = hessian_cache[model, param, paramT]
                hessian[model, param, param] += damping[model] * max(1e-12, abs(hessian_cache[model, param, param]))
    
    @staticmethod
    @nb.cuda.jit()
    def gpu_damping(hessian, hessian_cache, damping, damping_min, damping_max, tau, initialized, improved) :
        nmodels, nparams, _ = hessian.shape
        model, param, paramT = nb.cuda.grid(3)
        if model < nmodels and not improved[model] and param < nparams and paramT < nparams :
            hessian[model, param, paramT] = hessian_cache[model, param, paramT] 
            if param == paramT :
                hessian[model, param, param] += damping[model] * max(1e-12, abs(hessian_cache[model, param, param]))

    @prop(cache=True)
    def damping(self) :
        if not self.cuda : return self.cpu_damping
        threads_per_block = 8, 8, 8
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            (self.nparameters2fit + threads_per_block[2] - 1) // threads_per_block[2],
            )
        return self.gpu_damping[blocks_per_grid, threads_per_block]



    # --- Cholesky Solve ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_solve(hessian, gradient, steps, ignore) :
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

    @staticmethod
    @nb.cuda.jit()
    def gpu_solve(hessian, gradient, steps, ignore) :
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

    @prop(cache=True)
    def solve(self) :
        if not self.cuda : return self.cpu_solve
        threads_per_block = 64
        blocks_per_grid = self.nmodels
        return self.gpu_solve[blocks_per_grid, threads_per_block]



    # --- Parameter change ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_paramchange(parameters, indices, steps, bounds_min, bounds_max, ignore) :
        nmodels, nparams = steps.shape
        for model in nb.prange(nmodels) :
            if ignore[model] : continue
            for param in range(nparams) :
                val = parameters[model, indices[param]] + steps[model, param]
                if val < bounds_min[param]:
                    val = bounds_min[param]
                    steps[model, param] = val - parameters[model, indices[param]]
                elif val > bounds_max[param]:
                    val = bounds_max[param]
                    steps[model, param] = val - parameters[model, indices[param]]
                parameters[model, indices[param]] = val

    @nb.cuda.jit()
    def gpu_paramchange(parameters, indices, steps, bounds_min, bounds_max, ignore) :
        nmodels, nparams = steps.shape
        model, param = nb.cuda.grid(2)
        if model < nmodels and not ignore[model] and param < nparams :
            val = parameters[model, indices[param]] + steps[model, param]
            if val < bounds_min[param]:
                val = bounds_min[param]
                steps[model, param] = val - parameters[model, indices[param]]
            elif val > bounds_max[param]:
                val = bounds_max[param]
                steps[model, param] = val - parameters[model, indices[param]]
            parameters[model, indices[param]] = val
            
    @prop(cache=True)
    def paramchange(self) :
        if not self.cuda : return self.cpu_paramchange
        threads_per_block = 16, 16
        blocks_per_grid = (
            (self.nmodels + threads_per_block[0] - 1) // threads_per_block[0],
            (self.nparameters2fit + threads_per_block[1] - 1) // threads_per_block[1],
            )
        return self.gpu_paramchange[blocks_per_grid, threads_per_block]



    # --- Improving ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_improving(rho_data, old_chi2, new_chi2, parameters, indices, steps, gradient, hessian, damping, nu, damping_max, damping_min, converged, improved, ignore) :
        nmodels, nparams = steps.shape
        for model in nb.prange(nmodels) :
            if improved[model] : continue
            # Predicted residuals
            pred = 0.0
            for param in range(nparams):
                pred += steps[model, param] * (-gradient[model, param] + damping[model] * max(1e-12, abs(hessian[model, param, param])) * steps[model, param])
            pred *= 0.5
            # Acutal residuals
            ared = old_chi2[model] - new_chi2[model]
            # Rho
            rho = ared / pred if pred > 1e-12 else -1.0
            rho = min(max(rho, -1e6), 1e6)
            rho_data[model] = rho
            # Better
            if (not ignore[model]) and (pred > 1e-12) and (ared > 0.0):
                improved[model] = True
                tmp = 1.0 - (2.0 * rho - 1.0) ** 3
                if tmp < 1.0 / 3.0: tmp = 1.0 / 3.0
                elif tmp > 10.0:    tmp = 10.0
                damping[model] *= tmp
                nu[model] = 2.0
                if damping[model] < damping_min :
                    damping[model] = damping_min
            # Worse or failed
            else :
                if not ignore[model] :
                    for param in range(nparams) :
                        parameters[model, indices[param]] -= steps[model, param]
                    new_chi2[model] = old_chi2[model]
                if damping[model] >= damping_max:
                    converged[model] = -1
                    improved[model] = True # Did not really improve but we give up
                else :
                    damping[model] *= nu[model]
                    nu[model] *= 2.0
                    if damping[model] > damping_max :
                        damping[model] = damping_max

    @staticmethod
    @nb.cuda.jit()
    def gpu_improving(old_chi2, new_chi2, parameters, indices, steps, gradient, hessian, damping, nu, damping_max, damping_min, converged, improved, ignore) :
        nmodels, nparams = steps.shape
        model = nb.cuda.grid(1)
        if model < nmodels and not improved[model] :
            # Predicted residuals
            pred = 0.0
            for param in range(nparams):
                pred += steps[model, param] * (-gradient[model, param] + damping[model] * max(1e-12, abs(hessian[model, param, param])) * steps[model, param])
            pred *= 0.5
            # Acutal residuals
            ared = old_chi2[model] - new_chi2[model]
            # Rho
            rho = ared / pred if pred > 1e-12 else -1.0
            rho = min(max(rho, -1e6), 1e6)
            # Better
            if (not ignore[model]) and (pred > 1e-12) and (ared > 0.0):
                improved[model] = True
                tmp = 1.0 - (2.0 * rho - 1.0) ** 3
                if tmp < 1.0 / 3.0: tmp = 1.0 / 3.0
                elif tmp > 10.0:    tmp = 10.0
                damping[model] *= tmp
                nu[model] = 2.0
                if damping[model] < damping_min :
                    damping[model] = damping_min
            # Worse or failed
            else :
                if not ignore[model] :
                    for param in range(nparams) :
                        parameters[model, indices[param]] -= steps[model, param]
                    new_chi2[model] = old_chi2[model]
                if damping[model] >= damping_max:
                    converged[model] = -1
                    improved[model] = True # Did not really improve but we give up
                else :
                    damping[model] *= nu[model]
                    nu[model] *= 2.0
                    if damping[model] > damping_max :
                        damping[model] = damping_max

    @prop(cache=True)
    def improving(self) :
        if not self.cuda : return self.cpu_improving
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return self.gpu_improving[blocks_per_grid, threads_per_block]
