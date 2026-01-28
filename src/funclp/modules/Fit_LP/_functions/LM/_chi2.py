#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import numba as nb



def chi2(self, out) :
    self.estimator.deviance(self.raw_data, self.model_data, weights=self.weights, out=self.deviance_data, ignore=self.converged)
    processing_unit = 'gpu' if self.cuda else 'cpu'
    deviance2chi2_function =  getattr(self, f'{processing_unit}_deviance2chi2', None)
    if deviance2chi2_function is None :
        deviance2chi2_function = gpu_deviance2chi2(self) if self.cuda else cpu_deviance2chi2(self)
        setattr(self, f'{processing_unit}_deviance2chi2', deviance2chi2_function)
    deviance2chi2_function(self.deviance_data, self.chi2_data, self.converged)



def cpu_deviance2chi2(self) :
    @nb.njit(parallel=True)
    def func(deviance, chi2, converged) :
        nmodels, npoints = deviance.shape
        for model in nb.prange(nmodels) :
            if converged[model] : continue
            s = 0.0
            for point in range(npoints) :
                s += deviance[model, point]
            chi2[model] = s
    return func



def gpu_deviance2chi2(self) :
    @nb.cuda.jit()
    def func(deviance, chi2, converged) :
        nmodels, npoints = deviance.shape
        model = cuda.grid(1)  # 1D grid of threads
        if model < nmodels and not converged[model] :
            s = 0.0
            for point in range(npoints):
                s += deviance[model, point]
            chi2[model] = s
    threads_per_block = 128
    blocks_per_grid = (self.nmodels + threads_per_block - 1) // threads_per_block
    return func[blocks_per_grid, threads_per_block]
