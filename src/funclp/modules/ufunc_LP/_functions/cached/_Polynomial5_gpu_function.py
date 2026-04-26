
from ._Polynomial5_gpukernel_function import _Polynomial5_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Polynomial5_gpu_function(x, a, b, c, d, e, f, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], a[model], b[model], c[model], d[model], e[model], f[model], )
