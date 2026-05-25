
from ._Polynomial1_gpukernel_function import _Polynomial1_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Polynomial1_gpu_function(x, a, b, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], a[model], b[model], )
