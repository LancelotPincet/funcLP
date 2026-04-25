#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from scipy.interpolate import RectBivariateSpline
from funclp import Function, Parameter, ufunc
from corelp import rfrom
bspline2d, bspline2d_dx, bspline2d_dy, get_mean, get_amp, get_offset = rfrom("._splines", "bspline2d", "bspline2d_dx", "bspline2d_dy", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mux(res, *vars) :
    return get_mean(res, vars[0])
def muy(res, *vars) :
    return get_mean(res, vars[1])
def amp(res, *vars) :
    return get_amp(res)
def offset(res, *vars) :
    return get_offset(res)

# %% Function

class Spline2D(Function):

    def __init__(self, model2interp=None, x2interp=None, y2interp=None, kx=3, ky=3, /, **kwargs):
        if kx > 5 or ky > 5:
            raise ValueError('Spline of order higher than 5 is not implemented')
        if model2interp is not None and x2interp is not None and y2interp is not None:
            kx, ky = int(kx), int(ky)

            # Flatten coordinate arrays — accepts any broadcast shape
            model2interp = np.asarray(model2interp)
            if model2interp.size == x2interp.size and model2interp.size == y2interp.size :
                x2interp = np.asarray(x2interp)[0:1, :]
                y2interp = np.asarray(y2interp)[:, 0:1]
            x2interp = np.asarray(x2interp).ravel()
            y2interp = np.asarray(y2interp).ravel()
            if model2interp.shape != (len(x2interp), len(y2interp)):
                raise ValueError(
                    f'model2interp must have shape (nx, ny) = '
                    f'({len(x2interp)}, {len(y2interp)}), got {model2interp.shape}'
                )

            spline = RectBivariateSpline(x2interp, y2interp, model2interp, kx=kx, ky=ky)
            tx, ty, c = spline.tck
            tx = tx.astype(np.float32)
            ty = ty.astype(np.float32)
            nx = len(tx) - kx - 1
            ny = len(ty) - ky - 1
            coeffs = c[:nx * ny].reshape(ny, nx).astype(np.float32)

            kwargs.update(dict(kx=kx, ky=ky, tx=tx, ty=ty, coeffs=coeffs))

        super().__init__(**kwargs)

    @ufunc(
        variables=["x", "y"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("kx", 3),
            Parameter("ky", 3),
        ],
        constants=["tx", "ty", "coeffs"],
    )
    def function(x, y, /, mux=0., muy=0., amp=1., offset=0., kx=3, ky=3, tx=None, ty=None, coeffs=None) :
        return amp * bspline2d(tx, ty, coeffs, kx, ky, x-mux, y-muy) + offset

    @ufunc(constants=["tx", "ty", "coeffs"])
    def d_mux(x, y, /, mux, muy, amp, offset, kx, ky, tx=None, ty=None, coeffs=None):
        return -amp * bspline2d_dx(tx, ty, coeffs, kx, ky, x - mux, y - muy)

    @ufunc(constants=["tx", "ty", "coeffs"])
    def d_muy(x, y, /, mux, muy, amp, offset, kx, ky, tx=None, ty=None, coeffs=None):
        return -amp * bspline2d_dy(tx, ty, coeffs, kx, ky, x - mux, y - muy)

    @ufunc(constants=["tx", "ty", "coeffs"])
    def d_amp(x, y, /, mux, muy, amp, offset, kx, ky, tx=None, ty=None, coeffs=None):
        return bspline2d(tx, ty, coeffs, kx, ky, x - mux, y - muy)

    @ufunc(constants=["tx", "ty", "coeffs"])
    def d_offset(x, y, /, mux, muy, amp, offset, kx, ky, tx=None, ty=None, coeffs=None):
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
        np.linspace(-1, 1, 1000).reshape((1000,1)),
    )
    parameters = dict()

    # Get interpolation
    from funclp import Gaussian2D
    gausfunction = Gaussian2D()
    model = gausfunction(*variables)

    # Plot function
    instance = Spline2D(model, *variables)
    plot(instance, debug_folder, (variables[0] + 0.5, variables[1] - 0.5), parameters)
