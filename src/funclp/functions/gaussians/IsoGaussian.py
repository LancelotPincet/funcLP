#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import scipy.special as sc
import math
from funclp import Function, ufunc
from corelp import rfrom
gausfunc, get_mean, get_std, get_amp, get_offset, correct_angle = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset", "correct_angle")



# %% Parameters

def mux(res, *args) -> (None, None) :
    return get_mean(res, args[0])
def muy(res, *args) -> (None, None) :
    return get_mean(res, args[1])
def sig(res, *args) -> (0, None) :
    return np.sqrt(get_std(res, args[0]) * get_std(res, args[1]))
def amp(res, *args) -> (None, None) :
    return get_amp(res)
def offset(res, *args) -> (None, None) :
    return get_offset(res)

# %% Function

class IsoGaussian(Function):

    @ufunc(main=True)
    def function(x, y, /, mux:mux=0, muy:muy=0, sig:sig=1/(2*np.pi), amp:amp=1, offset:offset=0, pixx=-1, pixy=-1, nsig=-1) :
        return amp * gausfunc(x, mux, sig, 1, 0, pixx, nsig) * gausfunc(y, muy, sig, 1, 0, pixy, nsig) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_mux(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux) / sig**2
    @ufunc()
    def d_muy(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * (y - muy) / sig**2
    @ufunc()
    def d_sig(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * ((x - mux)**2 + (y - muy)**2) / sig**3
    @ufunc()
    def d_amp(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return exx * exy
    @ufunc()
    def d_offset(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        return 1



    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * (2 * np.pi) * self.sig**2
    @integ.setter
    def integ(self,value) :
        self.amp = value / np.sqrt(2 * np.pi) / self.sig**2
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2)) **2
    @proba.setter
    def proba(self,value) :
        self.nsig = sc.erfinv(np.sqrt(value))*np.sqrt(2)
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
    @property
    def pix(self) :
        return np.sqrt(self.pixx * self.pixy)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy = value, value



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(0, 1, 1000).reshape((1,1000)),
        np.linspace(0, 1.5, 1000).reshape((1000,1)),
    )
    parameters = dict()

    # Plot function
    instance = IsoGaussian()
    plot(instance, debug_folder, variables, parameters)
