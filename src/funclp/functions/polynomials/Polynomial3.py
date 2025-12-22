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
    return 1
def b(res, *args) -> (None, None) :
    return 0
def c(res, *args) -> (None, None) :
    return 0
def d(res, *args) -> (None, None) :
    return 0



# %% Function

class Polynomial3(Function):

    @ufunc(main=True)
    def function(x, /, a:a=1, b:b=0, c:c=0, d:d=0) :
        return a * x**3 + b * x**2 + c * x + d
    
    

    # Parameters derivatives
    @ufunc()
    def d_a(x, /, a, b) :
        return x**3
    @ufunc()
    def d_b(x, /, a, b) :
        return x**2
    @ufunc()
    def d_c(x, /, a, b) :
        return x
    @ufunc()
    def d_d(x, /, a, b) :
        return 1



    # Other attributes
    
    @property
    def roots(self) :
        return np.roots([self.a,self.b,self.c,self.d])



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
    instance = Polynomial3()
    plot(instance, debug_folder, variables, parameters)
