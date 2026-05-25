
from ._Exponential1_gpukernel_d_tau import _Exponential1_gpukernel_d_tau as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Exponential1_gpu_d_tau(t, tau, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(t[point], tau[model], amp[model], offset[model], )
