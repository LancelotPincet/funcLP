
from ._Polynomial5_cpukernel_d_b import _Polynomial5_cpukernel_d_b as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Polynomial5_cpu_d_b(x, a, b, c, d, e, f, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], a[model], b[model], c[model], d[model], e[model], f[model], )
