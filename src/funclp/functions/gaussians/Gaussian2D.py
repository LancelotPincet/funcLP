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

def mux(res, *vars) -> (None, None) :
    return get_mean(res, vars[0])
def muy(res, *vars) -> (None, None) :
    return get_mean(res, vars[1])
def sigx(res, *vars) -> (0, None) :
    return get_std(res, vars[0])
def sigy(res, *vars) -> (0, None) :
    return get_std(res, vars[1])
def amp(res, *vars) -> (None, None) :
    return get_amp(res)
def offset(res, *vars) -> (None, None) :
    return get_offset(res)

# %% Function

class Gaussian2D(Function):

    @ufunc(main=True)
    def function(x, y, /, mux:mux=0, muy:muy=0, sigx:sigx=1/(2*np.pi), sigy:sigy=1/(2*np.pi), amp:amp=1, offset:offset=0, pixx=-1, pixy=-1, nsig=-1, theta=0) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        return amp * gausfunc(x, mux, sigx, 1, 0, pixx, nsig) * gausfunc(y, muy, sigy, 1, 0, pixy, nsig) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_mux(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        if theta == 0 :
            return amp * exx * exy * (x - mux) / sigx**2
        return amp * exx * exy * (math.cos(theta) * x / sigx**2 + math.sin(theta) * y / sigy**2)
    @ufunc()
    def d_muy(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        if theta == 0 :
            return amp * exx * exy * (y - muy) / sigy**2
        return amp * exx * exy * (-math.sin(theta) * x / sigx**2 + math.cos(theta) * y / sigy**2)
    @ufunc()
    def d_sigx(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux)**2 / sigx**3
    @ufunc()
    def d_sigy(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (y - muy)**2 / sigy**3
    @ufunc()
    def d_amp(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return exx * exy
    @ufunc()
    def d_offset(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        return 1
    @ufunc()
    def d_theta(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux) * (y - muy) * (1 / sigx**2 - 1 / sigy**2)



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
    def proba(self, value) :
        self.nsig = sc.erfinv(np.sqrt(value)) * np.sqrt(2)
    @property
    def w(self) :
        return 2 * self.sig
    @w.setter
    def w(self,value) :
        self.sig = value / 2
    @property
    def wx(self) :
        return 2 * self.sigx
    @wx.setter
    def wx(self,value) :
        self.sigx = value / 2
    @property
    def wy(self) :
        return 2 * self.sigy
    @wy.setter
    def wy(self,value) :
        self.sigy = value / 2
    @property
    def FWHM(self) :
        return np.sqrt(2 * np.log(2)) * self.w
    @FWHM.setter
    def FWHM(self,value) :
        self.w = value / np.sqrt(2 * np.log(2))
    @property
    def FWHMx(self) :
        return np.sqrt(2 * np.log(2)) * self.wx
    @FWHMx.setter
    def FWHMx(self,value) :
        self.wx = value / np.sqrt(2 * np.log(2))
    @property
    def FWHMy(self) :
        return np.sqrt(2 * np.log(2)) * self.wy
    @FWHMy.setter
    def FWHMy(self,value) :
        self.wy = value / np.sqrt(2 * np.log(2))
    @property
    def pix(self) :
        return np.sqrt(self.pixx * self.pixy)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy = value, value
    @property
    def sig(self) :
        return np.sqrt(self.sigx * self.sigy)
    @sig.setter
    def sig(self, value) :
        self.sigx, self.sigy = value, value



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
    instance = Gaussian2D()
    plot(instance, debug_folder, variables, parameters)
