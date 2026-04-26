from ._Spline3D_cpukernel_d_mux import _Spline3D_cpukernel_d_mux as d_mux
from ._Spline3D_cpukernel_d_muy import _Spline3D_cpukernel_d_muy as d_muy
from ._Spline3D_cpukernel_d_muz import _Spline3D_cpukernel_d_muz as d_muz
from ._Spline3D_cpukernel_d_amp import _Spline3D_cpukernel_d_amp as d_amp
from ._Spline3D_cpukernel_d_offset import _Spline3D_cpukernel_d_offset as d_offset
from ._Spline3D_cpukernel_d_kx import _Spline3D_cpukernel_d_kx as d_kx
from ._Spline3D_cpukernel_d_ky import _Spline3D_cpukernel_d_ky as d_ky
from ._Spline3D_cpukernel_d_kz import _Spline3D_cpukernel_d_kz as d_kz

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Spline3D_cpu_jacobian(x, y, z, mux, muy, muz, amp, offset, kx, ky, kz, tx, ty, tz, coeffs, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_mux(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_muy(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_muz(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_amp(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_offset(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[5] :
                jacobian[model, point, count] = d_kx(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[6] :
                jacobian[model, point, count] = d_ky(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1

            if bool2fit[7] :
                jacobian[model, point, count] = d_kz(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs)
                count += 1
