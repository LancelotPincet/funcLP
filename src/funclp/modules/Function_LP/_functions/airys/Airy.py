#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from funclp import Function, Parameter, ufunc
from corelp import rfrom
j1, get_mean, get_amp, get_offset = rfrom("._airys", "j1", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mu(res, *args) :
    return get_mean(res, args[0])
def amp(res, *args) :
    return get_amp(res)
def offset(res, *args) :
    return get_offset(res)



# %% Function

class Airy(Function):

    @ufunc(
        variables=["x"],
        parameters=[
            Parameter("mu", 0., estimate=mu),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("wl", 550.),
            Parameter("NA", 1.5),
            Parameter("tol", 1.),
        ],
    )
    def function(x, /, mu=0., amp=1., offset=0., wl=550., NA=1.5, tol=1.) :
        r = abs(x-mu)
        if r < tol :
            return amp + offset
        z = 2 * np.pi * r * NA / wl
        return amp * 2 * j1(z) / z + offset
    
    

    # Other attributes
    
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



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-1000, 1000, 1000),
    )
    parameters = dict()

    # Plot function
    instance = Airy()
    plot(instance, debug_folder, variables, parameters)
