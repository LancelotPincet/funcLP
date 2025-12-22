#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-19
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Estimator

"""
Class defining an estimator function for fitting data.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference
from abc import ABC, abstractmethod



# %% Class
class Estimator(ABC, CudaReference) :
    '''
    Class defining an estimator function for fitting data.
    
    Parameters
    ----------
    distribution : Distribution
        Distribution instance.
    kwargs : dict
        Attributes to change.

    Attributes
    ----------
    deviance : method
        How well the model fits the data
    loss : method
        Loss for gradient descent
    observed : method
        Observed Hessian (negative second derivative)
    fisher : method
        Expected Hessian (Fisher information)
    '''

    @prop()
    def name(self) :
        return self.__class__.__name__
    def __init__(self, distribution=None, **kwargs) :
        self.distribution = distribution
        self.distribution.cuda_reference = self
        selfkwargs(self, kwargs)

    #ABC
    @abstractmethod
    def deviance(self, raw_data, model_data, /, weights=1, **kwargs) :
        pass
    @abstractmethod
    def loss(self, raw_data, model_data, /, weights=1, **kwargs) :
        pass
    @abstractmethod
    def observed(self, raw_data, model_data, /, weights=1, **kwargs) :
        pass
    @abstractmethod
    def fisher(self, raw_data, model_data, /, weights=1, **kwargs) :
        pass



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)