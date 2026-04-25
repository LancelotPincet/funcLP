#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import math
import numpy as np
from funclp import Function, Parameter, ufunc
from corelp import rfrom
j1, get_mean, get_amp, get_offset = rfrom("._airys", "j1", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mux(res, *args) :
    return get_mean(res, args[0])
def muy(res, *args) :
    return get_mean(res, args[1])
def amp(res, *args) :
    return get_amp(res)
def offset(res, *args) :
    return get_offset(res)



# %% Function

class Airy2D(Function):

    @ufunc(
        variables=["x", "y"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("wl", 550.),
            Parameter("NA", 1.5),
            Parameter("tol", 1.),
        ],
    )
    def function(x, y, /, mux=0., muy=0., amp=1., offset=0., wl=550., NA=1.5, tol=1.) :
        r = math.sqrt((x - mux)**2 + (y - muy)**2)
        if r < tol :
            return amp + offset
        z = 2 * np.pi * r * NA / wl
        return amp * 2 * j1(z) / z + offset
    
    

    # Parameters derivatives
    @property
    def radius(self) :
        return 0.61*self.wl/self.NA
    @property
    def diameter(self) :
        return 1.22*self.wl/self.NA
    @property
    def FWHM(self) :
        return 0.51*self.wl/self.NA
    @property
    def sigma(self) :
        return 0.21*self.wl/self.NA
    @property
    def Rayleigh(self) :
        return 0.61*self.wl/self.NA
    @property
    def Sparrow(self) :
        return 0.47*self.wl/self.NA
    @property
    def Abbe(self) :
        return 0.5*self.wl/self.NA
    n = 1.33 #optical index [default=water]
    @property
    def Abbe_z(self) :
        return 2*self.n*self.wl/self.NA**2
    def psf(self,*args,**kwargs) :
        return self(*args,**kwargs)**2



    # Other attributes
    
    @property
    def attribute(self) :
        return None
    @attribute.setter
    def attribute(self, value) :
        pass



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-1000, 1000, 1000).reshape(1, 1000),
        np.linspace(-1000, 1000, 1000).reshape(1000, 1),
    )
    parameters = dict()

    # Plot function
    instance = Airy2D()
    plot(instance, debug_folder, variables, parameters)
