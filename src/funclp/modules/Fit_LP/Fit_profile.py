#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Fit

"""
Class defining fitting algorithms.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference, use_inputs, use_shapes, use_cuda, use_broadcasting
from abc import ABC, abstractmethod
import numpy as np
import numba as nb
import os
import time



# %% Class
class Fit(ABC, CudaReference) :
    '''
    Class defining fitting algorithms.
    
    Parameters
    ----------
    kwargs : dict
        Attributes to change.

    Attributes
    ----------
    function : Function
        function instance to fit
    estimator : Estimator
        estimator instance to use to fit
    fit : method
        Core function used defining the fitting algorithm.
    max_iterations : int
        Maximum number of loops in the algorithm used.
    '''

    @prop()
    def name(self) :
        return self.__class__.__name__
    def __init__(self, function, estimator, **kwargs) :
        self._function = function # Function object
        self._estimator = estimator # Estimator object
        self.cuda_reference = self.function
        self.estimator.cuda_reference = self.function
        selfkwargs(self, kwargs)
        

    # Function and estimator
    @property
    def function(self) :
        return self._function
    @property
    def estimator(self) :
        return self._estimator

    # Attributs
    max_iterations = 200 #Maximum number of iterations
    max_retries = 10 # Maximum of step retries for a given step change
    ftol = 1e-8 # Loop stops when chi2 change is lower than ftol.
    xtol = 1e-8 # Loop stops when parameter step is lower than xtol.
    gtol = 0 # Loop stops when gradient maximum is lower than gtol.
    profile = False # Prints timing breakdown when enabled.
    profile_label = ""

    def _profile_reset(self):
        enabled = bool(getattr(self, "profile", False))
        if not enabled:
            env = os.environ.get("FUNCLP_PROFILE", "").strip().lower()
            enabled = env in {"1", "true", "yes", "on"}
        self._profile_enabled = enabled
        self._profile_times = {}
        self._profile_start = time.perf_counter() if enabled else 0.0
        self._profile_probe_done = False

    def _profile_add(self, key, elapsed):
        if not self._profile_enabled:
            return
        self._profile_times[key] = self._profile_times.get(key, 0.0) + float(elapsed)

    def _profile_report(self):
        if not self._profile_enabled:
            return
        total = max(0.0, time.perf_counter() - self._profile_start)
        label = self.profile_label if self.profile_label else f"{self.__class__.__name__}:{self.function.__class__.__name__}"
        print(f"[funclp-profile] {label} total={total:.6f}s")
        tracked = 0.0
        for key, value in sorted(self._profile_times.items(), key=lambda item: item[1], reverse=True):
            tracked += value
            pct = (100.0 * value / total) if total > 0.0 else 0.0
            print(f"[funclp-profile]   {key}: {value:.6f}s ({pct:5.1f}%)")
        untracked = max(0.0, total - tracked)
        if untracked > 0.0:
            pct = (100.0 * untracked / total) if total > 0.0 else 0.0
            print(f"[funclp-profile]   untracked: {untracked:.6f}s ({pct:5.1f}%)")

    def _profile_sync_cuda(self):
        if not self._profile_enabled or not getattr(self, "cuda", False):
            return
        try:
            self.xp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    def _profile_probe_assembly(self):
        if not self._profile_enabled or not getattr(self, "cuda", False) or self._profile_probe_done:
            return
        # Assembly probes moved to Function implementations if needed.
        self._profile_probe_done = True

    # Parameters to fit
    @property
    def index2fit(self) :
        parameters2fit = self.parameters2fit
        return [i for i, param in enumerate(self.function.parameters.keys()) if param in parameters2fit]
    @property
    def bool2fit(self) :
        return [getattr(self.function, f'{param}_fit') for param in self.function.parameters.keys()]
    @property
    def parameters2fit(self) :
        return [key for key in self.function.parameters.keys() if getattr(self.function, f'{key}_fit')]
    @property
    def nparameters2fit(self) :
        n = len(self.parameters2fit)
        assert n <= 8, "Cannot fit more than 8 parameters"
        return n
    @property
    def lower_bounds(self) :
        bounds = [self.xp.asarray(getattr(self.function, f'{key}_min')) for key in self.parameters2fit]
        if all(bound.ndim == 0 for bound in bounds):
            return self.xp.asarray(bounds)

        normalized = []
        for bound in bounds:
            if bound.ndim == 0:
                normalized.append(self.xp.full(self.nmodels, bound, dtype=self.dtype))
            else:
                normalized.append(bound.reshape(self.nmodels).astype(self.dtype, copy=False))
        return self.xp.stack(normalized, axis=1)
    @property
    def upper_bounds(self) :
        bounds = [self.xp.asarray(getattr(self.function, f'{key}_max')) for key in self.parameters2fit]
        if all(bound.ndim == 0 for bound in bounds):
            return self.xp.asarray(bounds)

        normalized = []
        for bound in bounds:
            if bound.ndim == 0:
                normalized.append(self.xp.full(self.nmodels, bound, dtype=self.dtype))
            else:
                normalized.append(bound.reshape(self.nmodels).astype(self.dtype, copy=False))
        return self.xp.stack(normalized, axis=1)


    #ABC
    @abstractmethod
    def fit_init(self) :
        ''' Fit initialization specific to optimizer '''
        pass
    @abstractmethod
    def fit_optimize(self) :
        ''' Fit main logic optimizer '''
        pass
    def __call__(self, raw_data, *args, weights=np.float32(1.)) :
        ''' Fitting function '''
        self._profile_reset()
        t_step = time.perf_counter()

        # Start
        cache_cuda = self.cuda
        if hasattr(self.function, 'prepare_fit_inputs'):
            raw_data, args, weights = self.function.prepare_fit_inputs(raw_data, args, weights)
            fit_ufunc = self.function.fit_ufunc
        else:
            fit_ufunc = self.function.__class__.function
        inputs = use_inputs(fit_ufunc, args, self.function.parameters) # variables, data, parameters
        (nomodel, nopoint), (self.nmodels, self.npoints), in_shapes = use_shapes(*inputs) # (nomodel, nopoint), (nmodels, npoints), (variables_shapes, data_shapes, parameters_shapes)
        self.cuda, self.xp, transfer_back, blocks_per_grid, threads_per_block = use_cuda(self.function, (self.nmodels, self.npoints), inputs)
        self.variables, self.data, self.parameters, self.dtype = use_broadcasting(self.xp, *inputs, *in_shapes, (self.nmodels, self.npoints))
        self.parameters = self.xp.asarray(self.parameters)
        self.constants = [self.xp.asarray(arr) for arr in self.function.constants.values()]
        self.jitted = self.function.gpu_function[blocks_per_grid, threads_per_block] if self.cuda else self.function.cpu_function
        self.parameters_indices = self.xp.asarray(self.index2fit)
        self.parameters_bools = self.xp.asarray(self.bool2fit)
        self.bounds_min = self.lower_bounds
        self.bounds_max = self.upper_bounds
        self.assembly_kernel = self.function.get_assembly(self.estimator, self.cuda, self.nmodels)
        weights = self.xp.asarray(weights)
        self._profile_add("setup", time.perf_counter() - t_step)

      # Allocate memory
        t_step = time.perf_counter()
        self.raw_data = self.xp.asarray(raw_data).reshape((self.nmodels, self.npoints)) # Data to fit
        self.weights = weights if weights.size > 1 else self.xp.full(shape=(self.nmodels, self.npoints), fill_value=weights) # weights vector
        self.parameters_steps = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype) # Step to apply on each parameter
        self.chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype) # chi2 current vector
        self.chi2_cache = self.xp.empty_like(self.chi2_data) # chi2 reference vector to define improvement
        self.gradient_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype) # gradient matrix
        self.hessian_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype) # hessian matrix
        self.hessian_cache = self.xp.empty_like(self.hessian_data) # hessian cache
        self.converged = self.xp.zeros(shape=self.nmodels, dtype=self.xp.int8) # -4: reserved for parameter clamped to bounds (used by downstream apps), -3: optimization fail, -2: optimization terminated and failed, -1: optimization termination to test, 0: not converged yet, 1: gtol (gradient), 2: ftol (chi2), 3: xtol (steps)
        self._profile_add("allocate", time.perf_counter() - t_step)


        # Initialize
        t_step = time.perf_counter()
        self.fit_init()
        self._profile_add("fit_init", time.perf_counter() - t_step)

        self._profile_probe_assembly()

        # Iterations
        for _ in range(self.max_iterations) :

            # chi2, gradient, hessian
            t_step = time.perf_counter()
            self.assembly_kernel(self.raw_data, *self.variables, *self.data, *self.parameters, *self.constants, self.weights, self.chi2_data, self.gradient_data, self.hessian_data, self.parameters_bools, self.converged)
            self._profile_sync_cuda()
            self._profile_add("assembly", time.perf_counter() - t_step)

            # Reset caches
            t_step = time.perf_counter()
            self.hessian_cache[:] = self.hessian_data
            self.chi2_cache[:] = self.chi2_data
            self._profile_sync_cuda()
            self._profile_add("cache_copy", time.perf_counter() - t_step)

            # Fit main logic [depends on optimizer]
            self.fit_optimize()

            # Calculate convergence
            t_step = time.perf_counter()
            self.convergence(self.ftol, self.chi2_cache, self.chi2_data, self.gtol, self.gradient_data, self.xtol, self.parameters.T, self.parameters_indices, self.parameters_steps, self.converged)
            self._profile_sync_cuda()
            self._profile_add("convergence", time.perf_counter() - t_step)

            t_step = time.perf_counter()
            all_converged = self.converged.all()
            self._profile_sync_cuda()
            self._profile_add("converged_check", time.perf_counter() - t_step)
            if all_converged :
                break
        # End
        t_step = time.perf_counter()
        if transfer_back :
            self.parameters = [self.xp.asnumpy(param) for param in self.parameters]
            self.converged = self.xp.asnumpy(self.converged)
        if nomodel : self.parameters = [param.item() for param in self.parameters]
        self.cuda = cache_cuda
        self.function.parameters = {key: self.parameters[pos] if getattr(self.function, f'{key}_fit') else self.function.parameters[key] for pos, key in enumerate(self.function.parameters.keys())}
        self._profile_add("finalize", time.perf_counter() - t_step)
        self._profile_report()
        return {key: self.function.parameters[key] for key in self.parameters2fit}



    # --- Convergence ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_convergence(ftol, old_chi2, new_chi2, gtol, gradient, xtol, parameters, indices, parameters_steps, converged) :
        nmodels, nparams = gradient.shape
        for model in nb.prange(nmodels) :
            if converged[model] > 0 or converged[model] < -1 : continue # test if 0 or -1
            # gtol
            if gtol:
                maxi = 0.0
                for param in range(nparams) :
                    g = abs(gradient[model, param])
                    if maxi < g : maxi = g
                if maxi <= gtol :
                    converged[model] = 1
                    continue
            # ftol
            if ftol and converged[model] == 0: # Do not test on -1 status
                if abs(new_chi2[model] - old_chi2[model]) <= ftol * max(1.0, abs(old_chi2[model])) :
                    converged[model] = 2
                    continue
            # xtol
            if xtol:
                small = True
                for param in range(nparams):
                    x = parameters[model, indices[param]]
                    p = parameters_steps[model, param]
                    if abs(p) > xtol * (abs(x) + 1.0):
                        small = False
                        break
                if small:
                    converged[model] = 3
                    continue
            # Failure
            if converged[model] == -1 :
                converged[model] = -2



    @staticmethod
    @nb.cuda.jit(cache=True)
    def gpu_convergence(ftol, old_chi2, new_chi2, gtol, gradient, xtol, parameters, indices, parameters_steps, converged) :
        model = nb.cuda.grid(1)  # 1D grid of threads
        nmodels, nparams = gradient.shape
        if model >=nmodels or converged[model] > 0 or converged[model] < -1 : return # test if 0 or -1
        # gtol
        if gtol:
            maxi = 0.0
            for param in range(nparams) :
                g = abs(gradient[model, param])
                if maxi < g : maxi = g
            if maxi <= gtol :
                converged[model] = 1
                return
        # ftol
        if ftol and converged[model] == 0: # Do not test on -1 status
            if abs(new_chi2[model] - old_chi2[model]) <= ftol * max(1.0, abs(old_chi2[model])) :
                converged[model] = 2
                return
        # xtol
        if xtol:
            small = True
            for param in range(nparams):
                x = parameters[model, indices[param]]
                p = parameters_steps[model, param]
                if abs(p) > xtol * (abs(x) + 1.0):
                    small = False
                    break
            if small:
                converged[model] = 3
                return
        # Failure
        if converged[model] == -1 :
            converged[model] = -2
    


    @prop(cache=True)
    def convergence(self) :
        if not self.cuda : return self.cpu_convergence
        threads_per_block = 32
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return self.gpu_convergence[blocks_per_grid, threads_per_block]



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)
