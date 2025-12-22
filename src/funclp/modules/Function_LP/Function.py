#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Function

"""
Function class defining a function model.
"""



# %% Libraries
from corelp import prop, selfkwargs
from funclp import CudaReference
from abc import ABC, abstractmethod
import numpy as np



# %% Class
class Function(ABC, CudaReference) :
    '''
    Function class defining a function model.
    
    Parameters
    ----------
    kwargs : dict
        Attributes to change.

    Attributes
    ----------
    parameters : dict
        dictionnary with eah parameter current value(s)
    nmodels : int
        Number of models according to current parameters of the function.
    '''

    @prop()
    def name(self) :
        return self.__class__.__name__
    def __init__(self, **kwargs) :
        selfkwargs(self, kwargs)

    @abstractmethod
    def function(self) :
        pass
    def __call__(self, *args, **kwargs) :
        return self._function(*args, **kwargs)

    # Parameters
    @property
    def nmodels(self) :
        shape = np.broadcast_shapes(*[np.shape(getattr(self, param, [])) for param in self.parameters])
        if len(shape) > 1 :
            raise ValueError('Parameters cannot have more than 1 dimension')
        return shape[0] if len(shape) == 1 else 0



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)