
from ._Spline3D_cpukernel_d_kx import _Spline3D_cpukernel_d_kx as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Spline3D_cpu_d_kx(x, y, z, mux, muy, muz, amp, offset, kx, ky, kz, tx, ty, tz, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs, )
