
from ._Diamond_cpukernel_d_muy import _Diamond_cpukernel_d_muy as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Diamond_cpu_d_muy(x, y, d, mux, muy, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model], )
