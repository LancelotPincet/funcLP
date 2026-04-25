
from ._Airy2D_cpukernel_d_tol import _Airy2D_cpukernel_d_tol as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Airy2D_cpu_d_tol(x, y, mux, muy, amp, offset, wl, NA, tol, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], mux[model], muy[model], amp[model], offset[model], wl[model], NA[model], tol[model], )
