
from ._Spline2D_cpukernel_d_ky import _Spline2D_cpukernel_d_ky as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Spline2D_cpu_d_ky(x, y, mux, muy, amp, offset, kx, ky, tx, ty, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs, )
