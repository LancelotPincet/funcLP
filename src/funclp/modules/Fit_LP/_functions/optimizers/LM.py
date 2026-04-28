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
        self.damping_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype) # Damping data
        self.nu_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=2.0) # Daing factor value
        self.improved = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer loop improved --> Stop inner loop
        self.ignore = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer failed or improved --> continue to next inner loop
        self.damping_initialized = False

    def fit_optimize(self) :

        # Status
        self.xp.not_equal(self.converged, 0, out=self.improved) # Global converged are considered already improved in the inner loop

        # Looping several time on a given hessian with different dampings
        for _ in range(self.max_retries) :
            self.ignore[:] = self.improved # If already improved, we ignore from start

            # Damping, Cholseky solve, Parameter steps
            self.damped_step(self.hessian_data, self.hessian_cache, self.gradient_data, self.damping_data, self.damping_min, self.damping_max, self.damping_tau, self.damping_initialized, self.parameters.T, self.parameters_indices, self.parameters_steps, self.bounds_min, self.bounds_max, self.ignore)
            if not self.damping_initialized : self.damping_initialized = True
    
            # evaluate chi2 after step and check if improving
            self.trial_chi2(self.raw_data, *self.variables, *self.data, *self.parameters, *self.constants, self.weights, self.chi2_data, self.parameters.T, self.parameters_indices, self.parameters_steps, self.gradient_data, self.hessian_cache, self.damping_data, self.nu_data, self.damping_max, self.damping_min, self.converged, self.improved, self.ignore)
            
            # End loop
            if self.improved.all() :
                break
        self.converged[~self.improved] = -3



    # Damping step
        
    @staticmethod
    @nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
    def cpu_damped_step(hessian, hessian_cache, gradient, damping, damping_min, damping_max, tau, initialized, parameters, indices,  steps, bounds_min, bounds_max, ignore):
        nmodels, nparams, _ = hessian.shape

        for model in nb.prange(nmodels):
            if ignore[model]:
                continue

            # Damping

            # Initialization
            if not initialized :
                m = 0.0
                for p in range(nparams) :
                    v = max(1e-12, abs(hessian_cache[model, p, p]))
                    if v > m : m = v
                damping[model] = min(max(tau * m, damping_min), damping_max)

            # Build damped hessian : H = H0 + lambda * diag(H0)
            for i in range(nparams) :
                for j in range(nparams) :
                    hessian[model, i, j] = hessian_cache[model, i, j]
                hessian[model, i, i] += damping[model] * max(1e-12, abs(hessian_cache[model, i, i]))

            # Cholesky solve

            # lower-triangular part stores L
            failed = False
            for i in range(nparams):
                for j in range(i + 1):
                    s = hessian[model, i, j]
                    for k in range(j):
                        s -= hessian[model, i, k] * hessian[model, j, k]

                    if i == j:
                        if s <= 0.0:
                            failed = True
                            break
                        hessian[model, i, j] = math.sqrt(s)
                    else:
                        hessian[model, i, j] = s / hessian[model, j, j]

                if failed:
                    break

            # Non positive-definite Hessian: keep parameters unchanged and let
            # the trial stage increase damping.
            if failed:
                for p in range(nparams):
                    steps[model, p] = 0.0
                continue

            # Fowrward substitution: L y = -g
            for i in range(nparams):
                s = -gradient[model, i]
                for k in range(i):
                    s -= hessian[model, i, k] * steps[model, k]
                steps[model, i] = s / hessian[model, i, i]

            # Back substitution: L^T x = y
            for i in range(nparams - 1, -1, -1):
                s = steps[model, i]
                for k in range(i + 1, nparams):
                    s -= hessian[model, k, i] * steps[model, k]
                steps[model, i] = s / hessian[model, i, i]

            # Parameter steps

            # Bounded updates
            for p in range(nparams):
                pidx = indices[p]
                oldv = parameters[model, pidx]
                newv = oldv + steps[model, p]
                if newv < bounds_min[p]:
                    newv = bounds_min[p]
                    steps[model, p] = newv - oldv
                elif newv > bounds_max[p]:
                    newv = bounds_max[p]
                    steps[model, p] = newv - oldv
                parameters[model, pidx] = newv



    @staticmethod
    @nb.cuda.jit(cache=True)
    def gpu_damped_step(hessian, hessian_cache, gradient, damping, damping_min, damping_max, tau, initialized, parameters, indices, steps, bounds_min, bounds_max, ignore):
        model = nb.cuda.grid(1)

        nmodels, nparams, _ = hessian.shape
        if model >= nmodels or ignore[model] : return

        # DAMPING

        # Initialize damping once per model if needed
        if not initialized :
            m = 0.0
            for p in range(nparams) :
                v = max(1e-12, abs(hessian_cache[model, p, p]))
                if v > m : m = v
            damping[model] = min(max(tau * m, damping_min), damping_max)

        # Build damped Hessian: H = H0 + lambda * diag(H0)
        for i in range(nparams):
            for j in range(nparams):
                hessian[model, i, j] = hessian_cache[model, i, j]
            hessian[model, i, i] += damping[model] * max(1e-12, abs(hessian_cache[model, i, i]))
        
        # CHOLESKY SOLVE

        # Lower-triangular part stores L
        failed = False
        for i in range(nparams):
            for j in range(i + 1):
                s = hessian[model, i, j]
                for k in range(j):
                    s -= hessian[model, i, k] * hessian[model, j, k]
                if i == j:
                    if s <= 0.0:
                        failed = True
                        break
                    hessian[model, i, j] = math.sqrt(s)
                else:
                    hessian[model, i, j] = s / hessian[model, j, j]
            if failed:
                break

        if failed:
            for p in range(nparams):
                steps[model, p] = 0.0
            return

        # Forward substitution: L y = -g
        for i in range(nparams):
            s = -gradient[model, i]
            for k in range(i):
                s -= hessian[model, i, k] * steps[model, k]
            steps[model, i] = s / hessian[model, i, i]

        # Back substitution: L^T x = y
        for i in range(nparams - 1, -1, -1):
            s = steps[model, i]
            for k in range(i + 1, nparams):
                s -= hessian[model, k, i] * steps[model, k]
            steps[model, i] = s / hessian[model, i, i]

        # PARAM CHANGE

        # Apply bounded parameter update
        for p in range(nparams):
            pidx = indices[p]
            oldv = parameters[model, pidx]
            newv = oldv + steps[model, p]
            if newv < bounds_min[p]:
                newv = bounds_min[p]
                steps[model, p] = newv - oldv
            elif newv > bounds_max[p]:
                newv = bounds_max[p]
                steps[model, p] = newv - oldv
            parameters[model, pidx] = newv



    @property
    def damped_step(self) :
        if not self.cuda : return self.cpu_damped_step
        threads_per_block = 32
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return self.gpu_damped_step[blocks_per_grid, threads_per_block]



    # Chi2 trial

    @prop(cache=True)
    def cpu_trial_chi2(self):
        variables = [key for key in self.function.variables]
        data = [key for key in self.function.data]
        parameters = [key for key in self.function.parameters.keys()]
        constants = [key for key in self.function.constants]
        inputs = ', '.join(variables + data + parameters + constants)
        inputs_scalar = ', '.join([f'point_{key}' for key in variables] + [f'point_{key}' for key in data] + [f'model_{key}' for key in parameters] + constants)
        point_variables = '\n            '.join([f'point_{key} = {key}[point]' for key in variables])
        point_data = '\n            '.join([f'point_{key} = {key}[model, point]' for key in data])
        model_params = '\n        '.join([f'model_{key} = {key}[model]' for key in parameters])
        string = f'''
@nb.njit(parallel=True, nogil=True, fastmath=True)
def func(raw_data, {inputs}, weights, chi2, parameters, indices, steps, gradient, hessian, damping, nu, damping_max, damping_min, converged, improved, ignore):
    nmodels, npoints = raw_data.shape
    nparams = steps.shape[1]

    for model in nb.prange(nmodels):
        if ignore[model]: continue

        # Trial chi2

        chi_local = 0.0

        # Load model parameters once
        {model_params}

        for point in range(npoints):
            {point_variables}
            {point_data}
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar({inputs_scalar})
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            chi_local += dev

        new_chi2 = chi_local
        old_chi2 = chi2[model]

        # Improving logic

        # Predicted residuals
        pred = 0.0
        for param in range(nparams):
            pred += steps[model, param] * (-gradient[model, param] + damping[model] * max(1e-12, abs(hessian[model, param, param])) * steps[model, param])
        pred *= 0.5

        # Actual residuals
        ared = old_chi2 - new_chi2

        # Rho
        rho = ared / pred if pred > 1e-12 else -1.0
        rho = min(max(rho, -1e6), 1e6)

        # Better
        if (not ignore[model]) and (pred > 1e-12) and (ared > 0.0):
            improved[model] = True
            chi2[model] = new_chi2
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
            if damping[model] >= damping_max:
                converged[model] = -1
                improved[model] = True # Did not really improve but we give up
            else :
                damping[model] *= nu[model]
                nu[model] *= 2.0
                if damping[model] > damping_max :
                    damping[model] = damping_max
'''

        glob = {'nb': nb, 'model_scalar': self.function.cpukernel_function, 'deviance_scalar': self.estimator.cpukernel_deviance}
        loc = {}
        exec(string, glob, loc)
        return loc['func']



    @prop(cache=True)
    def gpu_trial_chi2(self):
        variables = [key for key in self.function.variables]
        data = [key for key in self.function.data]
        parameters = [key for key in self.function.parameters.keys()]
        constants = [key for key in self.function.constants]
        inputs = ', '.join(variables + data + parameters + constants)
        inputs_threads = ', '.join([f'thread_{key}' for key in variables] + [f'thread_{key}' for key in data] + [f'block_{key}' for key in parameters] + constants)
        thread_variables = '\n        '.join([f'thread_{key} = {key}[point]' for key in variables])
        thread_data = '\n        '.join([f'thread_{key} = {key}[model, point]' for key in data])
        block_params = '\n    '.join([f'block_{key} = {key}[model]' for key in parameters])

        string = f'''
@nb.cuda.jit()
def func(raw_data, {inputs}, weights, chi2, parameters, indices, steps, gradient, hessian, damping, nu, damping_max, damping_min, converged, improved, ignore):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = steps.shape[1]

    if model >= nmodels or ignore[model]: return

    # Trial chi2 accumulation

    chi_local = nb.float32(0.0)

    # Load model parameters once
    {block_params}

    # Loop over points
    for point in range(tid, npoints, bdim):
        {thread_variables}
        {thread_data}
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        # Scalar model + deviance
        pred = model_scalar({inputs_threads})
        dev = deviance_scalar(thread_raw_data, pred, thread_weight)
        chi_local += dev

    # Shared memory reduction
    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_chi[tid] = chi_local
    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        new_chi2 = s_chi[0]
        old_chi2 = chi2[model]
    
    # Improving logic

        # Predicted residuals
        pred = 0.0
        for param in range(nparams):
            step = steps[model, param]
            pred += step * (-gradient[model, param] + damping[model] * max(1e-12, abs(hessian[model, param, param])) * step)
        pred *= 0.5

        # Acutal residuals
        ared = old_chi2 - new_chi2

        # Rho
        rho = ared / pred if pred > 1e-12 else -1.0
        rho = min(max(rho, -1e6), 1e6)

        # Better
        if (not ignore[model]) and (pred > 1e-12) and (ared > 0.0):
            improved[model] = True
            chi2[model] = new_chi2
            tmp = 1.0 - (2.0 * rho - 1.0) ** 3
            if tmp < 1.0 / 3.0: tmp = 1.0 / 3.0
            elif tmp > 10.0: tmp = 10.0
            damping[model] *= tmp
            nu[model] = 2.0
            if damping[model] < damping_min:
                damping[model] = damping_min

        # Worse or failed
        else:
            if not ignore[model]:
                for param in range(nparams):
                    parameters[model, indices[param]] -= steps[model, param]
            if damping[model] >= damping_max:
                converged[model] = -1
                improved[model] = True
            else:
                damping[model] *= nu[model]
                nu[model] *= 2.0
                if damping[model] > damping_max:
                    damping[model] = damping_max
'''
        glob = {'nb': nb, 'TPB': 128, 'model_scalar': self.function.gpukernel_function,'deviance_scalar': self.estimator.gpukernel_deviance}
        loc = {}
        exec(string, glob, loc)
        return loc['func']



    @property
    def trial_chi2(self) :
        if not self.cuda : return self.cpu_trial_chi2
        threads_per_block = 128
        blocks_per_grid = self.nmodels
        return self.gpu_trial_chi2[blocks_per_grid, threads_per_block]
