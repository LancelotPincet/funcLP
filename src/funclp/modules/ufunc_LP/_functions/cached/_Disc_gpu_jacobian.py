from ._Disc_gpukernel_d_r import _Disc_gpukernel_d_r as d_r
from ._Disc_gpukernel_d_mux import _Disc_gpukernel_d_mux as d_mux
from ._Disc_gpukernel_d_muy import _Disc_gpukernel_d_muy as d_muy
from ._Disc_gpukernel_d_amp import _Disc_gpukernel_d_amp as d_amp
from ._Disc_gpukernel_d_offset import _Disc_gpukernel_d_offset as d_offset

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Disc_gpu_jacobian(x, y, r, mux, muy, amp, offset, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_r(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_mux(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_muy(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_amp(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_offset(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model])
            count += 1
