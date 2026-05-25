
from ._MyClass_gpukernel_nodata import _MyClass_gpukernel_nodata as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _MyClass_gpu_nodata(x, y, a, b, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], a[model], b[model], )
