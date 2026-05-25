
from ._Polynomial4_gpukernel_d_e import _Polynomial4_gpukernel_d_e as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Polynomial4_gpu_d_e(x, a, b, c, d, e, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], a[model], b[model], c[model], d[model], e[model], )
