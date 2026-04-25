#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Function, Parameter, ufunc
from corelp import rfrom
func1, func2 = rfrom("._file", "func1", "func2")



# %% Parameters

def myparam(res, *args) :
    return func1(res, *args)



# %% Function

class MyFunc(Function):

    @ufunc(
        variables=["variable"],
        parameters=[Parameter("parameter", 0., estimate=myparam)],
    )
    def function(variable, /, parameter=0.) :
        return None
    
    

    # Parameters derivatives
    @ufunc()
    def d_parameter(variable, /, parameter=0.) :
        return None



    # Other attributes
    
    @property
    def attribute(self) :
        return None
    @attribute.setter
    def attribute(self, value) :
        pass



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = ()
    parameters = dict()

    # Plot function
    instance = MyFunc() #TODO
    plot(instance, debug_folder, variables, parameters)
