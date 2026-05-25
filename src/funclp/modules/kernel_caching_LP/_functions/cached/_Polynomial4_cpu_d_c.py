
from ._Polynomial4_cpukernel_d_c import _Polynomial4_cpukernel_d_c as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Polynomial4_cpu_d_c(x, a, b, c, d, e, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], a[model], b[model], c[model], d[model], e[model], )
