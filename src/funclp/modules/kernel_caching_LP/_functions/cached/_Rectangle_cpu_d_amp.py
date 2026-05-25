
from ._Rectangle_cpukernel_d_amp import _Rectangle_cpukernel_d_amp as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Rectangle_cpu_d_amp(x, y, l, ratio, mux, muy, amp, offset, theta, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model], )
