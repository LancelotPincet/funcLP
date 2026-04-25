#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import math
from funclp import Function, Parameter, ufunc
from corelp import rfrom
get_r, get_amp, get_offset, get_mean = rfrom("._masks", "get_r", "get_amp", "get_offset", "get_mean")



# %% Parameters

def r(res, *args) :
    return get_r(res, args[0])
def mux(res, *args) :
    return get_mean(res, args[0])
def muy(res, *args) :
    return get_mean(res, args[1])
def amp(res, *args) :
    return get_amp(res)
def offset(res, *args) :
    return get_offset(res)



# %% Function

class Disc(Function):

    @ufunc(
        variables=["x", "y"],
        parameters=[
            Parameter("r", 1., estimate=r, bounds=(0, None)),
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
        ],
    )
    def function(x, y, /, r=1., mux=0., muy=0., amp=1., offset=0.) :
        return amp * (math.sqrt((x-mux)**2 + (y-muy)**2) <= r) + offset
    
    

# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-3, 3, 1000).reshape((1, 1000)),
        np.linspace(-3, 3, 1000).reshape((1000, 1)),
    )
    parameters = dict()

    # Plot function
    instance = Disc()
    plot(instance, debug_folder, variables, parameters)
