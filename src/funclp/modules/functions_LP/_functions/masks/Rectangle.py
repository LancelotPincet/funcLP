#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import math
from funclp import Function, ufunc
from corelp import rfrom
get_r, get_amp, get_offset, get_mean = rfrom("._masks", "get_r", "get_amp", "get_offset", "get_mean")



# %% Parameters

def l(res, *args) -> (0, None) :
    return get_r(res, args[0])
def ratio(res, *args) -> (None, None) :
    return get_r(res, args[0]) / get_r(res, args[1])
def mux(res, *args) -> (None, None) :
    return get_mean(res, args[0])
def muy(res, *args) -> (None, None) :
    return get_mean(res, args[1])
def amp(res, *args) -> (None, None) :
    return get_amp(res)
def offset(res, *args) -> (None, None) :
    return get_offset(res)



# %% Function

class Rectangle(Function):

    @ufunc(main=True)
    def function(x, y, /, l:l=1, ratio:ratio=1, mux:mux=0, muy:muy=0, amp:amp=1, offset:offset=0, theta=0) :
        theta = -np.radians(theta)
        x, y = x * math.cos(theta) + y * math.sin(theta), y * math.cos(theta) - x * math.sin(theta)
        mux, muy = mux*math.cos(theta) + muy*math.sin(theta), muy*math.cos(theta) - mux*math.sin(theta)
        return amp * (abs(x-mux) < l/2) * (abs(y-muy) < l/ratio/2) + offset
    
    

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
    instance = Rectangle()
    plot(instance, debug_folder, variables, parameters)
