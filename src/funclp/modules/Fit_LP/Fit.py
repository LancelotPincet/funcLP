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
        self.function = function # Function object
        self.estimator = estimator # Estimator object
        self.cuda_reference = self.function
        self.estimator.cuda_reference, self.estimator_cuda_rederence = self.function, self.estimator.cuda_reference
        selfkwargs(self, kwargs)

    # Attributs
    max_iterations = 200 #Maximum number of iterations

    # Parameters
    @prop(link='function')
    def parameters(self) :
        return 'parameters'



    #ABC
    @abstractmethod
    def fit(self) :
        ''' Method defining the core algorithm behind the fitting method '''
        pass
    def __call__(self, raw_data, *args, weights=np.float32(1.)) :
        ''' Fitting function '''

        # Start
        cache_cuda = self.cuda
        inputs = use_inputs(self.function.__class__.function, args, self.function.parameters) # variables, data, parameters
        (nomodel, nopoint), (nmodels, npoints), in_shapes = use_shapes(*inputs) # (nomodel, nopoint), (nmodels, npoints), (variables_shapes, data_shapes, parameters_shapes)
        self.cuda, xp, transfer_back, blocks_per_grid, threads_per_block = use_cuda(self.function, (nmodels, npoints), inputs)
        self.variables, self.data, self.parameters, dtype = use_broadcasting(xp, *inputs, *in_shapes, (nmodels, npoints))
        self.raw_data = xp.asarray(raw_data).reshape((nmodels, npoints))
        self.weights = xp.asarray(weights)
        self.converged = xp.zeros(shape=nmodels, dtype=xp.bool_)

        # Algorithm fit
        self.fit()

        # End
        if transfer_back : self.parameters = {key: xp.asnumpy(value) for key, value in self.parameters.items()}
        if nomodel : self.parameters = {key: value.item() for key, value in self.parameters.items()}
        self.function.parameters = self.parameters
        self.cuda = cache_cuda
        return self.parameters



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)