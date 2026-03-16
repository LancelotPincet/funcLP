#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from scipy.interpolate import RectBivariateSpline, LSQBivariateSpline, InterpolatedUnivariateSpline
from funclp import Function, ufunc
from corelp import rfrom
bspline3d, get_mean, get_amp, get_offset = rfrom("._splines", "bspline3d", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mux(res, *vars) -> (None, None) :
    return get_mean(res, vars[0])
def muy(res, *vars) -> (None, None) :
    return get_mean(res, vars[1])
def muz(res, *vars) -> (None, None) :
    return get_mean(res, vars[2])
def amp(res, *vars) -> (None, None) :
    return get_amp(res)
def offset(res, *vars) -> (None, None) :
    return get_offset(res)

# %% Function

class Spline3D(Function):

    def __init__(self, model2interp=None, x2interp=None, y2interp=None, z2interp=None, kx=3, ky=3, kz=3, /, **kwargs):
        if kx > 5 or ky > 5 or kz > 5:
            raise ValueError('Spline of order higher than 5 is not implemented')
        if model2interp is not None and x2interp is not None and y2interp is not None and z2interp is not None:
            kx, ky, kz = int(kx), int(ky), int(kz)

            # Flatten coordinate arrays — accepts any shape (broadcastable inputs)
            model2interp = np.asarray(model2interp)
            if model2interp.size == x2interp.size and model2interp.size == y2interp.size :
                x2interp = np.asarray(x2interp)[0:1, 0:1, :]
                y2interp = np.asarray(y2interp)[0:1, :, 0:1]
                z2interp = np.asarray(z2interp)[:, 0:1, 0:1]
            x2interp = np.asarray(x2interp).ravel()
            y2interp = np.asarray(y2interp).ravel()
            z2interp = np.asarray(z2interp).ravel()
            nx, ny, nz = len(x2interp), len(y2interp), len(z2interp)

            # model2interp must be (nx, ny, nz) after squeezing broadcast dims
            if model2interp.shape != (nx, ny, nz):
                raise ValueError(
                    f'model2interp must have shape (nx, ny, nz) = '
                    f'({nx}, {ny}, {nz}), got {model2interp.shape}'
                )

            # Step 1: fit each z-slice (nx, ny) with RectBivariateSpline
            sp0 = RectBivariateSpline(x2interp, y2interp, model2interp[:, :, 0], kx=kx, ky=ky)
            tx, ty, _ = sp0.tck
            tx = tx.astype(np.float32)
            ty = ty.astype(np.float32)
            nx_b = len(tx) - kx - 1
            ny_b = len(ty) - ky - 1

            all_coeffs = np.zeros((nz, ny_b, nx_b), dtype=np.float32)
            all_coeffs[0] = sp0.tck[2][:nx_b * ny_b].reshape(ny_b, nx_b)
            for i in range(1, nz):
                spi = RectBivariateSpline(x2interp, y2interp, model2interp[:, :, i], kx=kx, ky=ky)
                all_coeffs[i] = spi.tck[2][:nx_b * ny_b].reshape(ny_b, nx_b)

            # Step 2: fit 1D splines along z for each (iy, ix) coefficient position
            sp_z0 = InterpolatedUnivariateSpline(z2interp, all_coeffs[:, 0, 0], k=kz)
            tz = sp_z0._eval_args[0].astype(np.float32)
            nz_b = len(tz) - kz - 1

            coeffs_cube = np.zeros((nz_b, ny_b, nx_b), dtype=np.float32)
            for iy in range(ny_b):
                for ix in range(nx_b):
                    sp_z = InterpolatedUnivariateSpline(z2interp, all_coeffs[:, iy, ix], k=kz)
                    coeffs_cube[:, iy, ix] = sp_z.get_coeffs()

            coeffs = coeffs_cube

        super().__init__(kx=kx, ky=ky, kz=kz, tx=tx, ty=ty, tz=tz, coeffs=coeffs)

    @ufunc(main=True, constants=["tx", "ty", "tz", "coeffs"])
    def function(x, y, z, /, mux:mux=0., muy:muy=0., muz:muz=0., amp:amp=1., offset:offset=0., kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None) :
        return amp * bspline3d(tx, ty, tz, coeffs, kx, ky, kz, x-mux, y-muy, z-muz) + offset
    


# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(-1, 1, 100).reshape((1,1,100)),
        np.linspace(-1, 1, 100).reshape((1,100,1)),
        np.linspace(-1, 1, 100).reshape((100,1,1)),
    )
    parameters = dict()

    # Get interpolation
    from funclp import Gaussian3D
    gausfunction = Gaussian3D()
    model = gausfunction(*variables)

    # Plot function
    instance = Spline3D(model, *variables)
    instance(variables[0] + 0.5, variables[1] - 0.5, variables[2], **parameters)
