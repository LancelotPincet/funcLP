from ._Spline2D_cpukernel_d_mux import _Spline2D_cpukernel_d_mux as d_mux
from ._Spline2D_cpukernel_d_muy import _Spline2D_cpukernel_d_muy as d_muy
from ._Spline2D_cpukernel_d_amp import _Spline2D_cpukernel_d_amp as d_amp
from ._Spline2D_cpukernel_d_offset import _Spline2D_cpukernel_d_offset as d_offset
from ._Spline2D_cpukernel_d_kx import _Spline2D_cpukernel_d_kx as d_kx
from ._Spline2D_cpukernel_d_ky import _Spline2D_cpukernel_d_ky as d_ky

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Spline2D_cpu_jacobian(x, y, mux, muy, amp, offset, kx, ky, tx, ty, coeffs, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_mux(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_muy(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_amp(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_offset(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_kx(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1

            if bool2fit[5] :
                jacobian[model, point, count] = d_ky(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs)
                count += 1
