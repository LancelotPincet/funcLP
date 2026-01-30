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
        self.estimator.cuda_reference, self.estimator_cuda_rederence = self.function, self.estimator.cuda_reference
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
    ftol = 1e-6 # Loop stops when chi2 change is lower than ftol.
    xtol = 0 # Loop stops when parameter step is lower than xtol.
    gtol = 0 # Loop stops when gradient maximum is lower than gtol.

    # Parameters to fit
    @property
    def parameters2fit(self) :
        return [key for key in self.function.parameters.keys() if getattr(self.function, f'{key}_fit')]
    @property
    def nparameters2fit(self) :
        return len(self.parameters2fit)



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

        # Allocate memory
        self.raw_data = self.xp.asarray(raw_data).reshape((self.nmodels, self.npoints)) # Data to fit
        self.model_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Calculated data from current model
        self.weights = self.xp.asarray(weights) # weights vector
        self.parameter_steps = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, 1), dtype=self.dtype) # Step to apply on each parameter
        self.improved = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer loop improved --> Stop inner loop
        self.failed = self.xp.zeros(shape=self.nmodels, dtype=self.xp.bool_) # Bool vector: True if the inner optimizer failed --> continue to next inner loop
        self.converged = self.xp.zeros(shape=self.nmodels, dtype=self.xp.uint8) # Int vector: -1: failed global convergence, 0: not converged yet, 1: gtol (gradient), 2: ftol (chi2), 3: xtol (steps)
        self.jacobian_data = self.xp.empty(shape=(self.nmodels, self.npoints, self.nparameters2fit), dtype=self.dtype) # Jacobian matrix
        self.deviance_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Deviance matrix to calculate chi2
        self.chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype) # chi2 current vector
        self.chi2_cache = self.xp.empty_like(self.chi2_data) # chi2 reference vector to define improvement
        self.loss_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # Loss matrix to calculate gradient
        self.gradient_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, 1), dtype=self.dtype) # gradient matrix
        self.fisher_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype) # fisher matrix to calculate hessian
        self.hessian_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype) # hessian matrix

        # Initialize
        self.chi2()
        self.fit_init()

        # Iterations
        for _ in range(self.max_iterations) :

            # Reset caches
            self.chi2_cache[:] = self.chi2_data
            self.xp.greater(self.converged, 0, out=self.improved)

            # Evaluate local model
            self.jacobian()
            self.gradient()
            self.hessian()

            # Fit main logic [depends on optimizer]
            self.fit_optimize()

            # Calculate convergence
            self.convergence()
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

    def chi2(self) :
        self.function(*self.variables, *self.data, **self.parameters, out=self.model_data, ignore=self.converged)
        self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.converged)
        deviance2chi2_function = self.gpu_deviance2chi2 if self.cuda else self.cpu_deviance2chi2
        deviance2chi2_function(self.deviance_data, self.chi2_data, self.converged)

    @prop(cache=True)
    def cpu_deviance2chi2(self) :
        @nb.njit(parallel=True)
        def func(deviance, chi2, ignore) :
            nmodels, npoints = deviance.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                s = 0.0
                for point in range(npoints) :
                    s += deviance[model, point]
                chi2[model] = s
        return func

    @prop(cache=True)
    def gpu_deviance2chi2(self) :
        @nb.cuda.jit()
        def func(deviance, chi2, ignore) :
            model = cuda.blockIdx.x
            tid = cuda.threadIdx.x
            npoints = deviance.shape[1]
            smem = cuda.shared.array(256, dtype=nb.float32)
            s = 0.0
            for i in range(tid, npoints, cuda.blockDim.x):
                s += deviance[model, i]
            smem[tid] = s
            cuda.syncthreads()
            # reduction
            stride = cuda.blockDim.x // 2
            while stride > 0:
                if tid < stride:
                    smem[tid] += smem[tid + stride]
                cuda.syncthreads()
                stride //= 2
            if tid == 0:
                chi2[model] = smem[0]
        threads_per_block = 256
        blocks_per_grid = self.nmodels
        return func[blocks_per_grid, threads_per_block]



    # --- Jacobian ---

    def jacobian(self) :
        jacobian_function = self.gpu_jacobian if self.cuda else self.cpu_jacobian
        jacobian_function(*self.variables, *self.data, *self.parameters.values(), self.jacobian_data, self.converged)

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
        kernels = {f"d_{parameter}": getattr(self.function, f'gpukernel_d_{parameter}') for parameter in self.parameters2fit}
        inputs = ''
        inputs += ', '.join(self.function.variables) + ', ' if len(self.function.variables) > 0 else ''
        inputs += ', '.join(self.function.data) + ', ' if len(self.function.data) > 0 else ''
        inputs += ', '.join(self.parameters.keys())
        string = f'''
    @nb.cuda.jit()
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



    # --- Gradient ---

    def gradient(self) :
        self.estimator.loss(self.raw_data, self.model_data, weights=self.weights, out=self.loss_data, ignore=self.converged)
        loss2gradient_function = self.gpu_loss2gradient if self.cuda else self.cpu_loss2gradient
        loss2gradient_function(self.jacobian_data, self.loss_data, self.gradient_data, self.converged)

    @prop(cache=True)
    def cpu_loss2gradient(self) :
        @nb.njit(parallel=True)
        def func(jacobian, loss, gradient, ignore) :
            nmodels, npoints, nparams = jacobian.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    s = 0.0
                    for point in range(npoints) :
                        s += jacobian[model, point, param] * loss[model, point]
                    gradient[model, param, 0] = s
        return func

    @prop(cache=True)
    def gpu_loss2gradient(self) :
        @nb.cuda.jit()
        def func(jacobian, loss, gradient, ignore) :
            nmodels, npoints, nparams = jacobian.shape
            model, param = nb.cuda.grid(2)
            if model < nmodels and not ignore[model] and param < nparams :
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

    def hessian(self) :
        self.estimator.fisher(self.raw_data, self.model_data, weights=self.weights, out=self.fisher_data, ignore=self.converged)
        fisher2hessian_function = self.gpu_fisher2hessian if self.cuda else self.cpu_fisher2hessian
        fisher2hessian_function(self.jacobian_data, self.fisher_data, self.hessian_data, self.converged)

    @prop(cache=True)
    def cpu_fisher2hessian(self) :
        @nb.njit(parallel=True)
        def func(jacobian, fisher, hessian, ignore) :
            nmodels, npoints, nparams = jacobian.shape
            for model in nb.prange(nmodels) :
                if ignore[model] : continue
                for param in range(nparams) :
                    for paramT in range(nparams) :
                        s = 0.0
                        for point in range(npoints) :
                            s += jacobian[model, point, param] * jacobian[model, point, paramT] * fisher[model, point]
                        hessian[model, param, paramT] = s
        return func

    @prop(cache=True)
    def gpu_fisher2hessian(self) :
        @nb.cuda.jit()
        def func(jacobian, fisher, hessian, ignore) :
            nmodels, npoints, nparams = jacobian.shape
            model, param, paramT = nb.cuda.grid(3)
            if model < nmodels and not ignore[model] and param < nparams and paramT < nparams :
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



    # --- Convergence ---

    def convergence(self) :
        convergence_function = self.gpu_convergence if self.cuda else self.cpu_convergence
        convergence_function(self.ftol, self.chi2_cache, self.chi2_data, self.gtol, self.gradient_data, self.xtol, self.parameter_steps, self.improved, self.converged)

    @prop(cache=True)
    def cpu_convergence(self) :
        @nb.njit(parallel=True)
        def func(ftol, old_chi2, new_chi2, gtol, gradient, xtol, parameter_steps, improved, ignore) :
            nmodels, nparams, _ = gradient.shape
            for model in nb.prange(nmodels) :
                if ignore[model] or not improved[model] : continue
                # gtol
                if gtol:
                    maxi = 0.0
                    for param in range(nparams) :
                        g = abs(gradient[model, param, 0])
                        if maxi < g : maxi = g
                    if maxi <= gtol :
                        ignore[model] = 1
                        continue
                # ftol
                if ftol:
                    if abs(new_chi2[model] - old_chi2[model]) <= ftol :
                        ignore[model] = 2
                        continue
                # xtol
                if xtol:
                    sum = 0.0
                    for param in range(nparams) :
                        sum += parameter_steps[model, param, 0]**2
                    if sum <= xtol :
                        ignore[model] = 3
                        continue
        return func

    @prop(cache=True)
    def gpu_convergence(self) :
        @nb.cuda.jit()
        def func(ftol, old_chi2, new_chi2, gtol, gradient, xtol, parameter_steps, improved, ignore) :
            model = nb.cuda.grid(1)  # 1D grid of threads
            nmodels, nparams, _ = gradient.shape
            if model < nmodels and not ignore[model] and improved[model]:
                # gtol
                if gtol:
                    maxi = 0.0
                    for param in range(nparams) :
                        g = abs(gradient[model, param, 0])
                        if maxi < g : maxi = g
                    if maxi <= gtol :
                        ignore[model] = 1
                        return
                # ftol
                if ftol:
                    if abs(new_chi2[model] - old_chi2[model]) <= ftol :
                        ignore[model] = 2
                        return
                # xtol
                if xtol:
                    sum = 0.0
                    for param in range(nparams) :
                        sum += parameter_steps[model, param, 0]**2
                    if sum <= xtol :
                        ignore[model] = 3
                        return
        threads_per_block = 128
        blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
        return func[blocks_per_grid, threads_per_block]



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)