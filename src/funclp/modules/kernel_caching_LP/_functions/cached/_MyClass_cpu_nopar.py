
from ._MyClass_cpukernel_nopar import _MyClass_cpukernel_nopar as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _MyClass_cpu_nopar(x, y, constant, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], constant[model, point], )
