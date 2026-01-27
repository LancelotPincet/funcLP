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
    return np.nanmean(np.gradient(res,args[0]))/2
def b(res, *args) -> (None, None) :
    return np.nanmean(np.gradient(res-a(res,args[0])*args[0]**2,args[0]))
def c(res, *args) -> (None, None) :
    return np.nanmean(res - a(res,args[0])*args[0]**2 - b*(res,args[0])*args[0])



# %% Function

class Polynomial2(Function):

    @ufunc(main=True)
    def function(x, /, a:a=1, b:b=0, c:c=0) :
        return a * x**2 + b * x + c
    
    

    # Parameters derivatives
    @ufunc()
    def d_a(x, /, a, b) :
        return x**2
    @ufunc()
    def d_b(x, /, a, b) :
        return x
    @ufunc()
    def d_c(x, /, a, b) :
        return 1



    # Other attributes
    
    @property
    def roots(self) :
        return np.roots([self.a,self.b,self.c])
    @property
    def delta(self) :
        return self.b**2 - 4* self.a *self.c



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
    instance = Polynomial2()
    plot(instance, debug_folder, variables, parameters)
