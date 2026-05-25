
from ._Airy2D_cpukernel_function import _Airy2D_cpukernel_function as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Airy2D_cpu_function(x, y, mux, muy, amp, offset, wl, NA, tol, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model], )
