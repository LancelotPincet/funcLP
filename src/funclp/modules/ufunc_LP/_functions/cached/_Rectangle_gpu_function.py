
from ._Rectangle_gpukernel_function import _Rectangle_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Rectangle_gpu_function(x, y, l, ratio, mux, muy, amp, offset, theta, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model], )
