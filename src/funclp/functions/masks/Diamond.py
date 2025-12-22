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

def d(res, *args) -> (0, None) :
    return get_r(res, args[0])
def mux(res, *args) -> (None, None) :
    return get_mean(res, args[0])
def muy(res, *args) -> (None, None) :
    return get_mean(res, args[1])
def amp(res, *args) -> (None, None) :
    return get_amp(res)
def offset(res, *args) -> (None, None) :
    return get_offset(res)



# %% Function

class Diamond(Function):

    @ufunc(main=True)
    def function(x, y, /, d:d=1, mux:mux=0, muy:muy=0, amp:amp=1, offset:offset=0) :
        r = math.sqrt((x - mux)**2 + (y - muy)**2)
        mask = r <= (d * math.sqrt((1 - 1 / (abs(y - muy) / abs(x - mux) + 1))**2 + (1 / (abs(y - muy) / abs(x - mux) + 1))**2))
        return amp * mask + offset
    
    

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
    instance = Diamond()
    plot(instance, debug_folder, variables, parameters)
