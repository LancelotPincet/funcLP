
from ._Diamond_gpukernel_d_offset import _Diamond_gpukernel_d_offset as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Diamond_gpu_d_offset(x, y, d, mux, muy, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model], )
