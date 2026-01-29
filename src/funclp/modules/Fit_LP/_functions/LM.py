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
        
    def fit_optimize(self) :

        # Reset cache
        self.damping_cache[:] = self.damping_data
        self.hessian_Hdiag()

        # Looping several time on a given hessian with different dampings
        for _ in range(self.max_retries) :

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
            if self.improved.all() :
                break



    # --- Hessian diagonal ---

    def hessian_Hdiag(self) :
        Hdiag_function = self.gpu_Hdiag if self.cuda else self.cpu_Hdiag
        Hdiag_function(self.hessian_data, self.hessian_diag, self.converged)

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
        damping_function(self.hessian_data, self.hessian_diag, self.damping_data, self.improved)

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



    # --- Cholesky Solve ---

    def solve(self) :
        solve_function = self.gpu_solve if self.cuda else self.cpu_solve
        convergence_function(self.hessian_data, self.gradient_data, self.parameter_steps, self.improved)

    @prop(cache=True)
    def cpu_solve(self) :
        @nb.njit(parallel=True)
        def func(hessian, gradient, steps, ignore) :
            nmodels, nparams, _ = hessian.shape
            for model in prange(nmodels):
                if ignore[model]: continue
                
                # --- Cholesky factorization: H = L Lᵀ (in place) ---
                for param in range(nparams):
                    for j in range(param + 1):
                        s = hessian[model, param, j]
                        for k in range(j):
                            s -= hessian[model, param, k] * hessian[model, j, k]

                        if i == j:
                            if s <= 0.0:
                                ignore[model] = True
                                break
                            hessian[model, i, j] = np.sqrt(s)
                        else:
                            hessian[model, i, j] = s / hessian[model, j, j]

                    if ignore[model]:
                        break

                if ignore[model]:
                    continue

                # --- Forward substitution: L y = -g ---
                for i in range(nparams):
                    s = -gradient[m, i, 0]
                    for k in range(i):
                        s -= hessian[m, i, k] * steps[m, k, 0]
                    steps[m, i, 0] = s / hessian[m, i, i]

                # --- Back substitution: Lᵀ x = y ---
                for i in range(nparams - 1, -1, -1):
                    s = steps[m, i, 0]
                    for k in range(i + 1, nparams):
                        s -= hessian[m, k, i] * steps[m, k, 0]
                    steps[m, i, 0] = s / hessian[m, i, i]
            
        return func

    @prop(cache=True)
    def cpu_sgpu_solveolve(self) :
        @nb.cuda.jit()
        def func(ftol, old_chi2, new_chi2, gtol, gradient, xtol, parameter_steps, improved, ignore) :
            model = nb.cuda.grid(1)  # 1D grid of threads
            nmodels, nparams, _ = gradient.shape

        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
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
        improving_function(self.chi2_cache, self.chi2_data, self.parameters, self.parameter_steps, self.damping_data, self.damping_increase, self.damping_decrease, self.damping_max, self.damping_min, self.converged, self.improved)

    @prop(cache=True)
    def cpu_improve(self) :
        @nb.njit(parallel=True)
        def func(old_chi2, new_chi2, parameters, parameter_steps, damping, damping_increase, damping_decrease, damping_max, damping_min, converged, ignore) :
            nmodels, nparams = parameters.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                if old_chi2 > new_chi2 : #Better
                    ignore[model] = True
                    damping[model] *= damping_decrease
                    if damping[model] < damping_min :
                        damping[model] = damping_min
                else : #Worse
                    for param in range(nparams) :
                        parameters[model, param] -= parameter_steps[model, param, 0]
                    if damping[model] == damping_max:
                        converged[model] = -1
                        ignore[model] = True
                    else :
                        damping[model] *= damping_increase
                        if damping[model] > damping_max :
                            damping[model] = damping_max
        return func

    @prop(cache=True)
    def gpu_improve(self) :
        @nb.cuda.jit()
        def func(old_chi2, new_chi2, parameters, parameter_steps, damping, damping_increase, damping_decrease, damping_max, damping_min, converged, ignore) :
            nmodels, nparams = parameters.shape
            model = nb.cuda.grid(1)
            if model < nmodels and not ignore[model] :
                if old_chi2 > new_chi2 : #Better
                    ignore[model] = True
                    damping[model] *= damping_decrease
                    if damping[model] < damping_min :
                        damping[model] = damping_min
                else : #Worse
                    for param in range(nparams) :
                        parameters[model, param] -= parameter_steps[model, param, 0]
                    if damping[model] == damping_max:
                        converged[model] = -1
                        ignore[model] = True
                    else :
                        damping[model] *= damping_increase
                        if damping[model] > damping_max :
                            damping[model] = damping_max
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]

