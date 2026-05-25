
from ._Airy2D_gpukernel_d_offset import _Airy2D_gpukernel_d_offset as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Airy2D_gpu_d_offset(x, y, mux, muy, amp, offset, wl, NA, tol, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model], )
