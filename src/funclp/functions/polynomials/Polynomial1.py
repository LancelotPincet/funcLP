#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from funclp import Function, ufunc



# %% Parameters

def a(res, *args) -> (None, None) :
    return np.nanmean(np.gradient(res,args[0]))
def b(res, *args) -> (None, None) :
    return np.nanmean(res - a(res,*args)*args[0])



# %% Function

class Polynomial1(Function):

    @ufunc(main=True)
    def function(x, /, a:a=1, b:b=0) :
        return a * x + b
    
    

    # Parameters derivatives
    @ufunc()
    def d_a(x, /, a, b) :
        return x
    @ufunc()
    def d_b(x, /, a, b) :
        return 1



    # Other attributes
    
    @property
    def roots(self) :
        return np.roots([self.a,self.b])



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-10, 10, 1000),
    )
    parameters = dict()

    # Plot function
    instance = Polynomial1()
    plot(instance, debug_folder, variables, parameters)
