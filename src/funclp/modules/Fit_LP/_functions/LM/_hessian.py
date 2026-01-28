#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import numba as nb



def hessian(self) :
    self.estimator.fisher(self.raw_data, self.model_data, weights=self.weights, out=self.fisher_data, ignore=self.converged)
    processing_unit = 'gpu' if self.cuda else 'cpu'
    fisher2ghessian_function =  getattr(self, f'{processing_unit}_fisher2ghessian', None)
    if fisher2ghessian_function is None :
        fisher2ghessian_function = gpu_fisher2ghessian(self) if self.cuda else cpu_fisher2ghessian(self)
        setattr(self, f'{processing_unit}_fisher2ghessian', fisher2ghessian_function)
    fisher2ghessian_function(self.jacobian_data, self.fisher_data, self.hessian_data, self.converged)


@prop(cache=True)
def cpu_fisher2ghessian(self) :
    @nb.njit(parallel=True)
    def func(jacobian, fisher, hessian, converged) :
        nmodels, npoints, nparams = jacobian.shape
        for model in nb.prange(nmodels) :
            if converged[model] : continue
            for param in range(nparams) :
                for paramT in range(nparams) :
                    s = 0.0
                    for point in range(npoints) :
                        s += jacobian[model, point, param] * jacobian[model, point, paramT] * fisher[model, point]
                    hessian[model, param, paramT] = s
    return func



@prop(cache=True)
def gpu_fisher2ghessian(self) :
    @nb.cuda.jit()
    def func(jacobian, fisher, hessian, converged) :
        nmodels, npoints, nparams = jacobian.shape
        model, param, paramT = cuda.grid(3)
        if model < nmodels and not converged[model] and param < nparams and paramT < nparams :
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
