#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-19
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Distribution

"""
Class defining the noise distribution in data.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference
from abc import ABC, abstractmethod
import numpy as np



# %% Class
class Distribution(ABC, CudaReference) :
    '''
    Class defining the noise distribution in data.
    
    Parameters
    ----------
    kwargs : dict
        Attributes to change.

    Attributes
    ----------
    pdf : method
        Probability Density Function for the given distribution
    loglikelihood_reduced : method
        Log-likelihood up to additive constants.
    loglikelihood : method
        Exact log-likelihood (with constants).
    dloglikelihood : method
        Derivative of log-likelihood w.r.t model parameter.
    d2loglikelihood : method
        Second derivative of log-likelihood (observed curvature).
    fisher : method
        Expected curvature (Fisher information).
    '''

    @prop()
    def name(self) :
        return self.__class__.__name__
    def __init__(self, **kwargs) :
        selfkwargs(self, kwargs)

    @abstractmethod
    def pdf(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(1.)
    def __call__(self, *args, **kwargs) :
        return self._pdf(*args, **kwargs)

    @abstractmethod
    def loglikelihood_reduced(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(0.)

    @abstractmethod
    def loglikelihood(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(0.)

    @abstractmethod
    def dloglikelihood(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(0.)

    @abstractmethod
    def d2loglikelihood(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(0.)

    @abstractmethod
    def fisher(self, raw_data, model_data, /, weights=np.float32(1.), ignore=np.bool_(False)):
        return np.float32(0.)
    


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)