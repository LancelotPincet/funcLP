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
gausfunc, get_mean, get_std, get_amp, get_offset, correct_angle_3D = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset", "correct_angle_3D")



# %% Parameters

def mux(res, *vars) -> (None, None) :
    return get_mean(res, vars[0])
def muy(res, *vars) -> (None, None) :
    return get_mean(res, vars[1])
def muz(res, *vars) -> (None, None) :
    return get_mean(res, vars[2])
def sigx(res, *vars) -> (0, None) :
    return get_std(res, vars[0])
def sigy(res, *vars) -> (0, None) :
    return get_std(res, vars[1])
def sigz(res, *vars) -> (0, None) :
    return get_std(res, vars[2])
def amp(res, *vars) -> (None, None) :
    return get_amp(res)
def offset(res, *vars) -> (None, None) :
    return get_offset(res)

# %% Function

class Gaussian3D(Function):

    @ufunc(main=True)
    def function(x, y, z, /, mux:mux=0., muy:muy=0., muz:muz=0., sigx:sigx=1/(2*np.pi)**(3/2), sigy:sigy=1/(2*np.pi)**(3/2), sigz:sigz=1/(2*np.pi)**(3/2), amp:amp=1., offset:offset=0., pixx=-1., pixy=-1., pixz=-1., nsig=-1., theta=0., phi=0.) :
        x, y, z, mux, muy, muz = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        return amp * gausfunc(x, mux, sigx, 1, 0, pixx, nsig) * gausfunc(y, muy, sigy, 1, 0, pixy, nsig) * gausfunc(z, muz, sigz, 1, 0, pixz, nsig) + offset
    
    

    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * (2 * np.pi)**(3/2) * self.sig**3 / np.abs(self.pix)**3
    @integ.setter
    def integ(self,value) :
        self.amp = value / (2 * np.pi)**(3/2) / self.sig**2 * np.abs(self.pix)**3
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2)) **3
    @proba.setter
    def proba(self, value) :
        self.nsig = sc.erfinv(value**(1/3)) * np.sqrt(2)
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
    def wz(self) :
        return 2 * self.sigz
    @wz.setter
    def wz(self,value) :
        self.sigz = value / 2
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
    def FWHMz(self) :
        return np.sqrt(2 * np.log(2)) * self.wz
    @FWHMz.setter
    def FWHMz(self,value) :
        self.wz = value / np.sqrt(2 * np.log(2))
    @property
    def pix(self) :
        return (self.pixx * self.pixy * self.pixz)**(1/3)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy, self.pixz = value, value, value
    @property
    def sig(self) :
        return (self.sigx * self.sigy * self.sigz)**(1/3)
    @sig.setter
    def sig(self, value) :
        self.sigx, self.sigy, self.sigz = value, value, value



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(0, 1, 100).reshape((1,1,100)),
        np.linspace(0, 1.5, 100).reshape((1,100,1)),
        np.linspace(0, 1., 100).reshape((100,1,1)),
    )
    parameters = dict()

    # Plot function
    instance = Gaussian3D()
    instance(*variables, **parameters)
