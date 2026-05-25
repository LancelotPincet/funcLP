
from ._MyClass_cpukernel_nodata import _MyClass_cpukernel_nodata as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _MyClass_cpu_nodata(x, y, a, b, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], a[model], b[model], )
