#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from funclp import Function, Parameter, ufunc



# %% Parameters

def a(res, *args) :
    return 1
def b(res, *args) :
    return 0
def c(res, *args) :
    return 0
def d(res, *args) :
    return 0
def e(res, *args) :
    return 0
def f(res, *args) :
    return 0



# %% Function

class Polynomial5(Function):

    @ufunc(
        variables=["x"],
        parameters=[
            Parameter("a", 1., estimate=a),
            Parameter("b", 0., estimate=b),
            Parameter("c", 0., estimate=c),
            Parameter("d", 0., estimate=d),
            Parameter("e", 0., estimate=e),
            Parameter("f", 0., estimate=f),
        ],
    )
    def function(x, /, a=1., b=0., c=0., d=0., e=0., f=0.) :
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    
    

    # Parameters derivatives
    @ufunc()
    def d_a(x, /, a, b, c, d, e, f) :
        return x**5
    @ufunc()
    def d_b(x, /, a, b, c, d, e, f) :
        return x**4
    @ufunc()
    def d_c(x, /, a, b, c, d, e, f) :
        return x**3
    @ufunc()
    def d_d(x, /, a, b, c, d, e, f) :
        return x**2
    @ufunc()
    def d_e(x, /, a, b, c, d, e, f) :
        return x
    @ufunc()
    def d_f(x, /, a, b, c, d, e, f) :
        return 1



    # Other attributes
    
    @property
    def roots(self) :
        return np.roots([self.a,self.b,self.c,self.d,self.e,self.f])



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
    instance = Polynomial5()
    plot(instance, debug_folder, variables, parameters)
