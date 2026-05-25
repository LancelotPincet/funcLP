
from ._Rectangle_cpukernel_function import _Rectangle_cpukernel_function as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Rectangle_cpu_function(x, y, l, ratio, mux, muy, amp, offset, theta, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model], )
