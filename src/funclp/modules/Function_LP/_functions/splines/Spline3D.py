#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
from scipy.interpolate import RectBivariateSpline, LSQBivariateSpline, InterpolatedUnivariateSpline
from funclp import Function, Parameter, ufunc
from corelp import rfrom
bspline3d, bspline3d_dx, bspline3d_dy, bspline3d_dz, get_mean, get_amp, get_offset = rfrom("._splines", "bspline3d", "bspline3d_dx", "bspline3d_dy", "bspline3d_dz", "get_mean", "get_amp", "get_offset")
bspline3d, get_mean, get_amp, get_offset = rfrom("._splines", "bspline3d", "get_mean", "get_amp", "get_offset")



# %% Parameters

def mux(res, *vars) :
    return get_mean(res, vars[0])
def muy(res, *vars) :
    return get_mean(res, vars[1])
def muz(res, *vars) :
    return get_mean(res, vars[2])
def amp(res, *vars) :
    return get_amp(res)
def offset(res, *vars) :
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

            kwargs.update(dict(tx=tx, ty=ty, tz=tz, coeffs=coeffs))

        kwargs.update(dict(kx=kx, ky=ky, kz=kz))
        super().__init__(**kwargs)

    @ufunc(
        variables=["x", "y", "z"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("muz", 0., estimate=muz),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("kx", 3),
            Parameter("ky", 3),
            Parameter("kz", 3),
        ],
        constants=["tx", "ty", "tz", "coeffs"],
    )
    def function(x, y, z, /, mux=0., muy=0., muz=0., amp=1., offset=0., kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None) :
        return amp * bspline3d(tx, ty, tz, coeffs, kx, ky, kz, x-mux, y-muy, z-muz) + offset
    
    @ufunc(constants=["tx", "ty", "tz", "coeffs"])
    def d_mux(x, y, z, /, mux, muy, muz, amp, offset, kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None):
        return -amp * bspline3d_dx(tx, ty, tz, coeffs, kx, ky, kz, x - mux, y - muy, z - muz)

    @ufunc(constants=["tx", "ty", "tz", "coeffs"])
    def d_muy(x, y, z, /, mux, muy, muz, amp, offset, kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None):
        return -amp * bspline3d_dy(tx, ty, tz, coeffs, kx, ky, kz, x - mux, y - muy, z - muz)

    @ufunc(constants=["tx", "ty", "tz", "coeffs"])
    def d_muz(x, y, z, /, mux, muy, muz, amp, offset, kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None):
        return -amp * bspline3d_dz(tx, ty, tz, coeffs, kx, ky, kz, x - mux, y - muy, z - muz)

    @ufunc(constants=["tx", "ty", "tz", "coeffs"])
    def d_amp(x, y, z, /, mux, muy, muz, amp, offset, kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None):
        return bspline3d(tx, ty, tz, coeffs, kx, ky, kz, x - mux, y - muy, z - muz)

    @ufunc(constants=["tx", "ty", "tz", "coeffs"])
    def d_offset(x, y, z, /, mux, muy, muz, amp, offset, kx=3, ky=3, kz=3, tx=None, ty=None, tz=None, coeffs=None):
        return np.float32(1.0)

    def _cpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return "from funclp.modules.Function_LP._functions.splines._splines import bspline3d, bspline3d_dx, bspline3d_dy, bspline3d_dz"

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return '''block_mux = mux[model]
        block_muy = muy[model]
        block_muz = muz[model]
        block_amp = amp[model]
        block_offset = offset[model]
        block_kx = kx[model]
        block_ky = ky[model]
        block_kz = kz[model]'''

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return '''            base = bspline3d(tx, ty, tz, coeffs, block_kx, block_ky, block_kz, point_x - block_mux, point_y - block_muy, point_z - block_muz)
            mod = block_amp * base + block_offset

            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '''            if bool2fit[0]:
                jacob_local[count] = -block_amp * bspline3d_dx(tx, ty, tz, coeffs, block_kx, block_ky, block_kz, point_x - block_mux, point_y - block_muy, point_z - block_muz)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = -block_amp * bspline3d_dy(tx, ty, tz, coeffs, block_kx, block_ky, block_kz, point_x - block_mux, point_y - block_muy, point_z - block_muz)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = -block_amp * bspline3d_dz(tx, ty, tz, coeffs, block_kx, block_ky, block_kz, point_x - block_mux, point_y - block_muy, point_z - block_muz)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = base
                count += 1
            if bool2fit[4]:
                jacob_local[count] = 1.0
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_kx(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
                count += 1
            if bool2fit[6]:
                jacob_local[count] = d_ky(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
                count += 1
            if bool2fit[7]:
                jacob_local[count] = d_kz(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
                count += 1'''

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return '''        if bool2fit[0]:
            jacob_local[count] = d_mux(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[1]:
            jacob_local[count] = d_muy(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[2]:
            jacob_local[count] = d_muz(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[3]:
            jacob_local[count] = d_amp(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[4]:
            jacob_local[count] = d_offset(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[5]:
            jacob_local[count] = d_kx(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[6]:
            jacob_local[count] = d_ky(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1
        if bool2fit[7]:
            jacob_local[count] = d_kz(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_amp, block_offset, block_kx, block_ky, block_kz, tx, ty, tz, coeffs)
            count += 1'''



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
