#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import math
import numpy as np
from funclp import Function, ufunc



# %% Parameters

def w0(res, *args) -> (0, None) :
    return np.nanmin(res)
def z0(res, *args) -> (None, None) :
    return args[0][np.nanargmin(res)]
def m2(res, *args) -> (1, None) :
    return 1



# %% Function

class GaussianBeam(Function):

    @ufunc(main=True)
    def function(z, /, w0:w0=10, z0:z0=0, m2:m2=1, wl=550, n=1) : # w0 : µm, z0 : mm, wl : nm
        return w0 * math.sqrt(1 + ((z - z0) / (math.pi * w0**2 * n / wl / m2))**2)
    
    

    # Parameters derivatives
    @ufunc()
    def d_w0(z, /, w0, z0, m2, wl, n) :
        sq = ((z - z0) * wl * m2 / math.pi / w0**2 / n)**2
        return math.sqrt(1 + sq) + w0 * math.sqrt(1 + sq) * sq
    @ufunc()
    def d_z0(z, /, w0, z0, m2, wl, n) :
        if z0 == z :
            return 0
        sq = ((z - z0) * wl * m2 / math.pi / w0**2 / n)**2
        return w0 / math.sqrt(1 + sq) * sq / (z0-z)
    @ufunc()
    def d_m2(z, /, w0, z0, m2, wl, n) :
        sq = ((z - z0) * wl * m2 / math.pi / w0**2 / n)**2
        return w0 / math.sqrt(1 + sq) * sq / m2
    @ufunc()
    def d_wl(z, /, w0, z0, m2, wl, n) :
        sq = ((z - z0) * wl * m2 / math.pi / w0**2 / n)**2
        return w0 / math.sqrt(1 + sq) * sq / wl
    @ufunc()
    def d_n(z, /, w0, z0, m2, wl, n) :
        sq = ((z - z0) * wl * m2 / math.pi / w0**2 / n)**2
        return w0 / math.sqrt(1 + sq) * sq * n



    # Other attributes
    
    @property
    def zr(self) -> float : #Rayleigh distance [mm]
        return self.w0**2 * np.pi * self.n /self.wl/self.m2 *1e6
    @zr.setter
    def zr(self, value) -> float :
        self.w0 = np.sqrt(abs(value)/np.pi/self.n*self.wl*self.m2*1e-6)
    def R(self,z) : #Wavefront curvature [1/mm]
        with np.errstate(divide='ignore'):
            return z*(1+(np.asarray(self.zr)/(z-self.z0))**2)
    @property
    def divergence(self) -> float : #divergence of beam [rad]
        return np.arctan(self.wl*1e-6*self.m2/np.pi/self.w0)
    @divergence.setter
    def divergence(self, value) -> float :
        with np.errstate(divide='ignore'):
            self.w0 = self.wavelength*1e-6/np.pi/np.tan(value)
    @property
    def NA(self) -> float : #numerical aperture of gaussian beam
        return self.w0*self.m2/self.zr



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
    instance = GaussianBeam()
    plot(instance, debug_folder, variables, parameters)
