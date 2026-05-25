from ._Airy2D_gpukernel_d_mux import _Airy2D_gpukernel_d_mux as d_mux
from ._Airy2D_gpukernel_d_muy import _Airy2D_gpukernel_d_muy as d_muy
from ._Airy2D_gpukernel_d_amp import _Airy2D_gpukernel_d_amp as d_amp
from ._Airy2D_gpukernel_d_offset import _Airy2D_gpukernel_d_offset as d_offset
from ._Airy2D_gpukernel_d_wl import _Airy2D_gpukernel_d_wl as d_wl
from ._Airy2D_gpukernel_d_NA import _Airy2D_gpukernel_d_NA as d_NA
from ._Airy2D_gpukernel_d_tol import _Airy2D_gpukernel_d_tol as d_tol

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Airy2D_gpu_jacobian(x, y, mux, muy, amp, offset, wl, NA, tol, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_mux(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_muy(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_amp(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_offset(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_wl(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_NA(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1

        if bool2fit[6] :
            jacobian[model, point, count] = d_tol(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model])
            count += 1
