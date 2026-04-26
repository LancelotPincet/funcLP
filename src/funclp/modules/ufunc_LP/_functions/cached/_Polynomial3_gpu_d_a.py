
from ._Polynomial3_gpukernel_d_a import _Polynomial3_gpukernel_d_a as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Polynomial3_gpu_d_a(x, a, b, c, d, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], a[model], b[model], c[model], d[model], )
