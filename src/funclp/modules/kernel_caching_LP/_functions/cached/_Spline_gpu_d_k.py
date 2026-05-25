
from ._Spline_gpukernel_d_k import _Spline_gpukernel_d_k as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Spline_gpu_d_k(x, mu, amp, offset, k, t, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs, )
