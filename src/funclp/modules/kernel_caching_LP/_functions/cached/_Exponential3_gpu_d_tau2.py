
from ._Exponential3_gpukernel_d_tau2 import _Exponential3_gpukernel_d_tau2 as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Exponential3_gpu_d_tau2(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model], )
