
from ._Exponential3_cpukernel_d_amp3 import _Exponential3_cpukernel_d_amp3 as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Exponential3_cpu_d_amp3(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(t[point], tau1[model], tau2[model], tau3[model], amp1[model], amp2[model], amp3[model], offset[model], )
