
from ._MyClass_cpukernel_novar import _MyClass_cpukernel_novar as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _MyClass_cpu_novar(constant, a, b, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(constant[model, point], a[model], b[model], )
