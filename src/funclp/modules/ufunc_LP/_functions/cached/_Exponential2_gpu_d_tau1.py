
from ._Exponential2_gpukernel_d_tau1 import _Exponential2_gpukernel_d_tau1 as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Exponential2_gpu_d_tau1(t, tau1, tau2, amp1, amp2, offset, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model], )
