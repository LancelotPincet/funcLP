#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import math
from funclp import Function, ufunc
from corelp import rfrom
get_tau, get_amp, get_offset = rfrom("._exponentials", "get_tau", "get_amp", "get_offset")



# %% Parameters

def tau(res, *args) -> (None, None) :
    return get_tau(res, args[0])
def amp(res, *args) -> (None, None) :
    return get_amp(res)
def offset(res, *args) -> (None, None) :
    return get_offset(res)



# %% Function

class Exponential1(Function):

    @ufunc(main=True)
    def function(t, /, tau:tau=1, amp:amp=1, offset:offset=0) :
        return amp * math.exp(-t / tau) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_tau(t, /, tau, amp, offset) :
        return amp * math.exp(-t / tau) * t / tau**2
    @ufunc()
    def d_amp(t, /, tau, amp, offset) :
        return math.exp(-t / tau)
    @ufunc()
    def d_offset(t, /, tau, amp, offset) :
        return 1



    # Other attributes
    
    @property
    def k(self) :
        return 1 / self.tau
    @k.setter
    def k(self,value) :
        self.tau = 1 / value
    @property
    def tau_half(self) :
        return self.tau * np.log(2)
    @tau_half.setter
    def tau_half(self,value) :
        self.tau = value / np.log(2)



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(0, 10, 1000),
    )
    parameters = dict()

    # Plot function
    instance = Exponential1()
    plot(instance, debug_folder, variables, parameters)
