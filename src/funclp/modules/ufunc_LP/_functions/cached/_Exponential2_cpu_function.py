
from ._Exponential2_cpukernel_function import _Exponential2_cpukernel_function as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Exponential2_cpu_function(t, tau1, tau2, amp1, amp2, offset, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model], )
