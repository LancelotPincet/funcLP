
from ._Disc_cpukernel_d_mux import _Disc_cpukernel_d_mux as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Disc_cpu_d_mux(x, y, r, mux, muy, amp, offset, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], r[model], mux[model], muy[model], amp[model], offset[model], )
