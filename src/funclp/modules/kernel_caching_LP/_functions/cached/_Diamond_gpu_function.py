
from ._Diamond_gpukernel_function import _Diamond_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Diamond_gpu_function(x, y, d, mux, muy, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model], )
