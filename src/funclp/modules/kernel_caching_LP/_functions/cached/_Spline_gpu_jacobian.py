from ._Spline_gpukernel_d_mu import _Spline_gpukernel_d_mu as d_mu
from ._Spline_gpukernel_d_amp import _Spline_gpukernel_d_amp as d_amp
from ._Spline_gpukernel_d_offset import _Spline_gpukernel_d_offset as d_offset
from ._Spline_gpukernel_d_k import _Spline_gpukernel_d_k as d_k

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Spline_gpu_jacobian(x, mu, amp, offset, k, t, coeffs, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_mu(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs)
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_amp(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs)
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_offset(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs)
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_k(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs)
            count += 1
