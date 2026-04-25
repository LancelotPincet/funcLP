#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from funclp import Function, Parameter, ufunc
from corelp import rfrom
bspline1d, bspline1d_dx, get_mean, get_amp, get_offset = rfrom("._splines", "bspline1d", "bspline1d_dx", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mu(res, *vars) :
    return get_mean(res, vars[0])
def amp(res, *vars) :
    return get_amp(res)
def offset(res, *vars) :
    return get_offset(res)

# %% Function

class Spline(Function):

    def __init__(self, model2interp=None, x2interp=None, k=3, /, **kwargs):
        if k > 5:
            raise ValueError('Spline of order higher than 5 is not implemented')
        if model2interp is not None and x2interp is not None:
            k = int(k)

            # Flatten coordinate arrays — accepts any broadcast shape
            x2interp = np.asarray(x2interp).ravel()
            model2interp = np.asarray(model2interp).ravel()
            if model2interp.shape != (len(x2interp),):
                raise ValueError(
                    f'model2interp must have shape (nx,) = '
                    f'({len(x2interp)},), got {model2interp.shape}'
                )

            spline = InterpolatedUnivariateSpline(x2interp, model2interp, k=k)
            # _eval_args[0] is the full knot vector (with k+1 boundary repetitions)
            # get_knots() returns interior only — do NOT use it here
            t      = spline._eval_args[0].astype(np.float32)
            # get_coeffs() already has length len(t)-k-1, correctly trimmed
            coeffs = spline.get_coeffs().astype(np.float32)

        super().__init__(k=k, t=t, coeffs=coeffs)

    @ufunc(
        variables=["x"],
        parameters=[
            Parameter("mu", 0., estimate=mu),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("k", 3),
        ],
        constants=["t", "coeffs"],
    )
    def function(x, /, mu=0., amp=1., offset=0., k=3, t=None, coeffs=None) :
        return amp * bspline1d(t, coeffs, k, x-mu) + offset
    
    @ufunc(constants=["t", "coeffs"])
    def d_mu(x, /, mu, amp, offset, k=3, t=None, coeffs=None):
        return -amp * bspline1d_dx(t, coeffs, k, x - mu)

    @ufunc(constants=["t", "coeffs"])
    def d_amp(x, /, mu, amp, offset, k=3, t=None, coeffs=None):
        return bspline1d(t, coeffs, k, x - mu)

    @ufunc(constants=["t", "coeffs"])
    def d_offset(x, /, mu, amp, offset, k=3, t=None, coeffs=None):
        return np.float32(1.0)  



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-1, 1, 1000).reshape((1,1000)),
    )
    parameters = dict()

    # Get interpolation
    from funclp import Gaussian
    gausfunction = Gaussian()
    model = gausfunction(*variables)

    # Plot function
    instance = Spline(model, *variables)
    plot(instance, debug_folder, (variables[0] + 0.5), parameters)
