
from ._Airy_gpukernel_d_amp import _Airy_gpukernel_d_amp as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Airy_gpu_d_amp(x, mu, amp, offset, wl, NA, tol, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model], )
