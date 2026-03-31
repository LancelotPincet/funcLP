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
        return len(self.parameters2fit)
    @property
    def lower_bounds(self) :
        return self.xp.asarray([getattr(self.function, f'{key}_min') for key in self.parameters2fit])
    @property
    def upper_bounds(self) :
        return self.xp.asarray([getattr(self.function, f'{key}_max') for key in self.parameters2fit])


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

        # Start
        cache_cuda = self.cuda
        inputs = use_inputs(self.function.__class__.function, args, self.function.parameters) # variables, data, parameters
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
        self.jacobian = self.function.gpu_jacobian[blocks_per_grid, threads_per_block]if self.cuda else self.function.cpu_jacobian

      # Allocate memory
        self.raw_data = self.xp.asarray(raw_data).reshape((self.nmodels, self.npoints)) # Data to fit
        self.model_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Calculated data from current model
        self.weights = self.xp.asarray(weights) # weights vector
        self.parameters_steps = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype) # Step to apply on each parameter
        self.jacobian_data = self.xp.empty(shape=(self.nmodels, self.npoints, self.nparameters2fit), dtype=self.dtype) # Jacobian matrix
        self.deviance_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Deviance matrix to calculate chi2
        self.chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype) # chi2 current vector
        self.chi2_cache = self.xp.empty_like(self.chi2_data) # chi2 reference vector to define improvement
        self.loss_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Loss matrix to calculate gradient
        self.gradient_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit), dtype=self.dtype) # gradient matrix
        self.fisher_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # fisher matrix to calculate hessian
        self.hessian_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype) # hessian matrix
        self.hessian_cache = self.xp.empty_like(self.hessian_data) # hessian cache
        self.converged = self.xp.zeros(shape=self.nmodels, dtype=self.xp.int8) # -3: optimization fail, -2: optimization terminated and failed, -1: optimization termination to test, 0: not converged yet, 1: gtol (gradient), 2: ftol (chi2), 3: xtol (steps)


        # Initialize
        self.fit_init()

        # Iterations
        for _ in range(self.max_iterations) :

            # Chi2
            self.jitted(*self.variables, *self.data, *self.parameters, *self.constants, self.model_data, self.converged)
            self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.converged)
            self.deviance2chi2(self.deviance_data, self.chi2_data, self.converged)

            # Jacobian
            self.jacobian(*self.variables, *self.data, *self.parameters, *self.constants, self.jacobian_data, self.parameters_bools, self.converged)

            # Gradient
            self.estimator.loss(self.raw_data, self.model_data, weights=self.weights, out=self.loss_data, ignore=self.converged)
            self.loss2gradient(self.jacobian_data, self.loss_data, self.gradient_data, self.converged)

            # Hessian
            self.estimator.fisher(self.raw_data, self.model_data, weights=self.weights, out=self.fisher_data, ignore=self.converged)
            self.fisher2hessian(self.jacobian_data, self.fisher_data, self.hessian_data, self.converged)

            # Reset caches
            self.hessian_cache[:] = self.hessian_data
            self.chi2_cache[:] = self.chi2_data

            # Fit main logic [depends on optimizer]
            self.fit_optimize()

            # Calculate convergence
            self.convergence(self.ftol, self.chi2_cache, self.chi2_data, self.gtol, self.gradient_data, self.xtol, self.parameters.T, self.parameters_indices, self.parameters_steps, self.converged)
            if self.converged.all() :
                break

        # End
        if transfer_back :
            self.parameters = [self.xp.asnumpy(param) for param in self.parameters]
            self.converged = self.xp.asnumpy(self.converged)
        if nomodel : self.parameters = [param.item() for param in self.parameters]
        self.cuda = cache_cuda
        self.function.parameters = {key: self.parameters[pos] if getattr(self.function, f'{key}_fit') else self.function.parameters[key] for pos, key in enumerate(self.function.parameters.keys())}
        return {key: self.function.parameters[key] for key in self.parameters2fit}



    # --- Chi² ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_deviance2chi2(deviance, chi2, ignore) :
        nmodels, npoints = deviance.shape
        for model in nb.prange(nmodels) :
            if ignore[model] : continue
            s = 0.0
            for point in range(npoints) :
                s += deviance[model, point]
            chi2[model] = s

    @staticmethod
    @nb.cuda.jit()
    def gpu_deviance2chi2(deviance, chi2, ignore) :
        model = nb.cuda.blockIdx.x
        tid = nb.cuda.threadIdx.x
        if ignore[model]:
            return
        npoints = deviance.shape[1]
        smem = nb.cuda.shared.array(256, dtype=nb.float32)
        s = 0.0
        for i in range(tid, npoints, nb.cuda.blockDim.x):
            s += deviance[model, i]
        smem[tid] = s
        nb.cuda.syncthreads()
        # reduction
        stride = nb.cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            nb.cuda.syncthreads()
            stride //= 2
        if tid == 0:
            chi2[model] = smem[0]

    @prop(cache=True)
    def deviance2chi2(self) :
        if not self.cuda : return self.cpu_deviance2chi2
        threads_per_block = 256
        blocks_per_grid = self.nmodels
        return self.gpu_deviance2chi2[blocks_per_grid, threads_per_block]



    # --- Gradient ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_loss2gradient(jacobian, loss, gradient, ignore) :
        nmodels, npoints, nparams = jacobian.shape
        for model in nb.prange(nmodels) :
            if ignore[model] : continue
            for param in range(nparams) :
                s = 0.0
                for point in range(npoints) :
                    s += jacobian[model, point, param] * loss[model, point]
                gradient[model, param] = s

    @staticmethod
    @nb.cuda.jit()
    def gpu_loss2gradient(jacobian, loss, gradient, ignore) :
        model = nb.cuda.blockIdx.x
        tid = nb.cuda.threadIdx.x
        if ignore[model] :
            return
        npoints = jacobian.shape[1]
        nparams  = jacobian.shape[2]
        g = nb.cuda.shared.array((64,), nb.float32)
        # init
        for j in range(tid, nparams, nb.cuda.blockDim.x):
            g[j] = 0.0
        nb.cuda.syncthreads()
        # accumulate
        for i in range(tid, npoints, nb.cuda.blockDim.x):
            li = loss[model, i]
            for j in range(nparams):
                nb.cuda.atomic.add(g, j, jacobian[model, i, j] * li)
        nb.cuda.syncthreads()
        # write back
        for j in range(tid, nparams, nb.cuda.blockDim.x):
            gradient[model, j] = g[j]
    
    @prop(cache=True)
    def loss2gradient(self) :
        if not self.cuda : return self.cpu_loss2gradient
        threads_per_block = 256
        blocks_per_grid = self.nmodels
        return self.gpu_loss2gradient[blocks_per_grid, threads_per_block]


    # --- Hessian ---

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def cpu_fisher2hessian(jacobian, fisher, hessian, ignore) :
        nmodels, npoints, nparams = jacobian.shape
        for model in nb.prange(nmodels) :
            if ignore[model] : continue
            for param in range(nparams) :
                for paramT in range(nparams) :
                    s = 0.0
                    for point in range(npoints) :
                        s += jacobian[model, point, param] * jacobian[model, point, paramT] * fisher[model, point]
                    hessian[model, param, paramT] = s

    @staticmethod
    @nb.cuda.jit()
    def gpu_fisher2hessian(jacobian, fisher, hessian, ignore):
        model = nb.cuda.blockIdx.x
        tid   = nb.cuda.threadIdx.x
        nt    = nb.cuda.blockDim.x
        if ignore[model]:
            return
        npoints = jacobian.shape[1]
        nparams = jacobian.shape[2]
        H = nb.cuda.shared.array((64, 64), nb.float32)
        # init
        for j in range(tid, nparams, nt):
            for k in range(j, nparams):
                H[j, k] = 0.0
        nb.cuda.syncthreads()
        # accumulation (each j owned by one thread)
        for i in range(tid, npoints, nt):
            fi = fisher[model, i]
            for j in range(tid, nparams, nt):
                Jij = jacobian[model, i, j]
                vj  = Jij * fi
                for k in range(j, nparams):
                    H[j, k] += vj * jacobian[model, i, k]
        nb.cuda.syncthreads()
        # write back
        for j in range(tid, nparams, nt):
            for k in range(j, nparams):
                v = H[j, k]
                hessian[model, j, k] = v
                hessian[model, k, j] = v
    
    @prop(cache=True)
    def fisher2hessian(self) :
        if not self.cuda : return self.cpu_fisher2hessian
        threads_per_block = 256
        blocks_per_grid = self.nmodels
        return self.gpu_fisher2hessian[blocks_per_grid, threads_per_block]



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
    @nb.cuda.jit()
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
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return self.gpu_convergence[blocks_per_grid, threads_per_block]



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)