#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import numba as nb



def gradient(self) :
    self.estimator.loss(self.raw_data, self.model_data, weights=self.weights, out=self.loss_data, ignore=self.converged)
    processing_unit = 'gpu' if self.cuda else 'cpu'
    loss2gradient_function =  getattr(self, f'{processing_unit}_loss2gradient', None)
    if loss2gradient_function is None :
        loss2gradient_function = gpu_loss2gradient(self) if self.cuda else cpu_loss2gradient(self)
        setattr(self, f'{processing_unit}_loss2gradient', loss2gradient_function)
    loss2gradient_function(self.jacobian_data, self.loss_data, self.gradient_data, self.converged)



def cpu_loss2gradient(self) :
    @nb.njit(parallel=True)
    def func(jacobian, loss, gradient, converged) :
        nmodels, npoints, nparams = jacobian.shape
        for model in nb.prange(nmodels) :
            if converged[model] : continue
            for param in range(nparams) :
                s = 0.0
                for point in range(npoints) :
                    s += jacobian[model, point, param] * loss[model, point]
                gradient[model, param, 0] = s
    return func



def gpu_loss2gradient(self) :
    @nb.cuda.jit()
    def func(jacobian, loss, gradient, converged) :
        nmodels, npoints, nparams = jacobian.shape
        model, param = cuda.grid(2)
        if model < nmodels and not converged[model] and param < nparams :
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
