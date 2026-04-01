#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Estimator
from corelp import prop
import numba as nb
from numba import cuda



# %% Function

class MLE(Estimator) :



    # Deviance

    def deviance(self, raw_data, model_data, weights=1, **kwargs) :
        ''' How well the model fits the data '''
        weights = (-2) * weights
        return self.distribution.loglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_deviance(self) :
        kernel = self.distribution.cpukernel_loglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.njit(nogil=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-2) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']

    @prop(cache=True)
    def gpukernel_deviance(self) :
        kernel = self.distribution.gpukernel_loglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.cuda.jit(device=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-2) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']



    # Loss

    def loss(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Loss for gradient descent '''
        weights = (-1) * weights
        return self.distribution.dloglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_loss(self) :
        kernel = self.distribution.cpukernel_dloglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.njit(nogil=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']

    @prop(cache=True)
    def gpukernel_loss(self) :
        kernel = self.distribution.gpukernel_dloglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.cuda.jit(device=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']



    # Observed

    def observed(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Observed Hessian (negative second derivative)'''
        weights = (-1) * weights
        return self.distribution.d2loglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_observed(self) :
        kernel = self.distribution.cpukernel_d2loglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.njit(nogil=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']

    @prop(cache=True)
    def gpukernel_observed(self) :
        kernel = self.distribution.gpukernel_d2loglikelihood
        default = self.distribution.default_attributes
        string = f'''
@nb.cuda.jit(device=True, fastmath=True)
def func(raw_data, model_data, weights) :
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']



    # Fisher

    def fisher(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Expected Hessian (Fisher information) '''
        return self.distribution.fisher(raw_data, model_data, weights, **kwargs)
    
    @prop(cache=True)
    def cpukernel_fisher(self) :
        kernel = self.distribution.cpukernel_fisher
        default = self.distribution.default_attributes
        string = f'''
@nb.njit(nogil=True, fastmath=True)
def func(raw_data, model_data, weights) :
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']

    @prop(cache=True)
    def gpukernel_fisher(self) :
        kernel = self.distribution.gpukernel_fisher
        default = self.distribution.default_attributes
        string = f'''
@nb.cuda.jit(device=True, fastmath=True)
def func(raw_data, model_data, weights) :
    return kernel(raw_data, model_data, weights, {default})
'''
        glob = {'nb': nb, 'kernel': kernel}
        loc = {}
        exec(string, glob, loc)
        return loc['func']


