#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import scipy.special as sc
from funclp import Function, ufunc
from corelp import rfrom
gausfunc, get_mean, get_std, get_amp, get_offset = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset")



# %% Parameters

def mu(res, *args) -> (None, None) :
    return get_mean(res, args[0])
def sig(res, *args) -> (0, None) :
    return get_std(res, args[0])
def amp(res, *args) -> (None, None) :
    return get_amp(res)
def offset(res, *args) -> (None, None) :
    return get_offset(res)



# %% Function

class Gaussian(Function):

    @ufunc(main=True)
    def function(x, /, mu:mu=0, sig:sig=1/np.sqrt(2*np.pi), amp:amp=1, offset:offset=0, pix=-1, nsig=-1) :
        return gausfunc(x, mu, sig, amp, offset, pix, nsig)
    


    # Parameters derivatives
    @ufunc()
    def d_mu(x, /, mu, sig, amp, offset, pix, nsig) :
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return amp * ex * (x - mu) / sig**2
    @ufunc()
    def d_sig(x, /, mu, sig, amp, offset, pix, nsig) :
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return amp * ex * (x - mu)**2 / sig**3
    @ufunc()
    def d_amp(x, /, mu, sig, amp, offset, pix, nsig) :
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return ex
    @ufunc()
    def d_offset(x, /, mu, sig, amp, offset, pix, nsig) :
        return 1



    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * np.sqrt(2 * np.pi) * self.sig
    @integ.setter
    def integ(self,value) :
        self.amp = value / np.sqrt(2 * np.pi) / self.sig
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2))
    @proba.setter
    def proba(self,value) :
        self.nsig = sc.erfinv(value) * np.sqrt(2)
    @property
    def w(self) :
        return 2 * self.sig
    @w.setter
    def w(self,value) :
        self.sig = value / 2
    @property
    def FWHM(self) :
        return np.sqrt(2 * np.log(2)) * self.w
    @FWHM.setter
    def FWHM(self,value) :
        self.w = value / np.sqrt(2 * np.log(2))



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs
    variables = (
        np.linspace(-1, 1, 1000),
    )
    parameters = dict()

    # Plot function
    instance = Gaussian()
    plot(instance, debug_folder, variables, parameters)
