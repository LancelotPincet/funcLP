
from ._Polynomial2_gpukernel_d_b import _Polynomial2_gpukernel_d_b as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Polynomial2_gpu_d_b(x, a, b, c, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], a[model], b[model], c[model], )
