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

def tau1(res, *args) -> (None, None) :
    n = int(len(res) / 2) + 1
    return get_tau(res[:n], args[0][:n])
def tau2(res, *args) -> (None, None) :
    n = int(len(res) / 2) + 1
    return get_tau(res[n:], args[0][n:])
def amp1(res, *args) -> (None, None) :
    return get_amp(res) / 2
def amp2(res, *args) -> (None, None) :
    return get_amp(res) / 2
def offset(res, *args) -> (None, None) :
    return get_offset(res)



# %% Function

class Exponential2(Function):

    @ufunc(main=True)
    def function(t, /, tau1:tau1=1, tau2:tau2=1/2, amp1:amp1=1/2, amp2:amp2=1/2, offset:offset=0) :
        return amp1 * math.exp(-t / tau1) + amp2 * math.exp(-t / tau2) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_tau1(t, /, tau1, tau2, amp1, amp2, offset) :
        return amp1 * math.exp(-t / tau1) * t / tau1**2
    @ufunc()
    def d_tau2(t, /, tau1, tau2, amp1, amp2, offset) :
        return amp2 * math.exp(-t / tau2) * t / tau2**2
    @ufunc()
    def d_amp1(t, /, tau1, tau2, amp1, amp2, offset) :
        return math.exp(-t / tau1)
    @ufunc()
    def d_amp2(t, /, tau1, tau2, amp1, amp2, offset) :
        return math.exp(-t / tau2)
    @ufunc()
    def d_offset(t, /, tau1, tau2, amp1, amp2, offset) :
        return 1



    # Other attributes
    
    @property
    def k1(self) :
        return 1 / self.tau1
    @k1.setter
    def k1(self,value) :
        self.tau1 = 1 / value
    @property
    def tau_half1(self) :
        return self.tau1 * np.log(2)
    @tau_half1.setter
    def tau_half1(self,value) :
        self.tau1 = value / np.log(2)
    @property
    def k2(self) :
        return 1 / self.tau2
    @k2.setter
    def k2(self,value) :
        self.tau2 = 1 / value
    @property
    def tau_half2(self) :
        return self.tau2 * np.log(2)
    @tau_half2.setter
    def tau_half2(self,value) :
        self.tau2 = value / np.log(2)



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
    instance = Exponential2()
    plot(instance, debug_folder, variables, parameters)
