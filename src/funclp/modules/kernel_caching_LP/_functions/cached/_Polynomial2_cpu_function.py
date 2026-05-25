
from ._Polynomial2_cpukernel_function import _Polynomial2_cpukernel_function as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Polynomial2_cpu_function(x, a, b, c, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], a[model], b[model], c[model], )
