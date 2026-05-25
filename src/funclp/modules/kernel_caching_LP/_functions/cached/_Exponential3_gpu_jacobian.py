from ._Exponential3_gpukernel_d_tau1 import _Exponential3_gpukernel_d_tau1 as d_tau1
from ._Exponential3_gpukernel_d_tau2 import _Exponential3_gpukernel_d_tau2 as d_tau2
from ._Exponential3_gpukernel_d_tau3 import _Exponential3_gpukernel_d_tau3 as d_tau3
from ._Exponential3_gpukernel_d_amp1 import _Exponential3_gpukernel_d_amp1 as d_amp1
from ._Exponential3_gpukernel_d_amp2 import _Exponential3_gpukernel_d_amp2 as d_amp2
from ._Exponential3_gpukernel_d_amp3 import _Exponential3_gpukernel_d_amp3 as d_amp3
from ._Exponential3_gpukernel_d_offset import _Exponential3_gpukernel_d_offset as d_offset

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Exponential3_gpu_jacobian(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_tau1(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_tau2(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_tau3(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_amp1(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_amp2(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_amp3(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1

        if bool2fit[6] :
            jacobian[model, point, count] = d_offset(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model])
            count += 1
