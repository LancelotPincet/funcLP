
from ._Exponential1_cpukernel_function import _Exponential1_cpukernel_function as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Exponential1_cpu_function(t, tau, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(t[point], tau[model], amp[model], offset[model], )
