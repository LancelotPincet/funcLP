from ._Rectangle_gpukernel_d_l import _Rectangle_gpukernel_d_l as d_l
from ._Rectangle_gpukernel_d_ratio import _Rectangle_gpukernel_d_ratio as d_ratio
from ._Rectangle_gpukernel_d_mux import _Rectangle_gpukernel_d_mux as d_mux
from ._Rectangle_gpukernel_d_muy import _Rectangle_gpukernel_d_muy as d_muy
from ._Rectangle_gpukernel_d_amp import _Rectangle_gpukernel_d_amp as d_amp
from ._Rectangle_gpukernel_d_offset import _Rectangle_gpukernel_d_offset as d_offset
from ._Rectangle_gpukernel_d_theta import _Rectangle_gpukernel_d_theta as d_theta

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Rectangle_gpu_jacobian(x, y, l, ratio, mux, muy, amp, offset, theta, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_l(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_ratio(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_mux(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_muy(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_amp(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_offset(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1

        if bool2fit[6] :
            jacobian[model, point, count] = d_theta(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model])
            count += 1
