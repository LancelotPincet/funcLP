from ._Airy_cpukernel_d_mu import _Airy_cpukernel_d_mu as d_mu
from ._Airy_cpukernel_d_amp import _Airy_cpukernel_d_amp as d_amp
from ._Airy_cpukernel_d_offset import _Airy_cpukernel_d_offset as d_offset
from ._Airy_cpukernel_d_wl import _Airy_cpukernel_d_wl as d_wl
from ._Airy_cpukernel_d_NA import _Airy_cpukernel_d_NA as d_NA
from ._Airy_cpukernel_d_tol import _Airy_cpukernel_d_tol as d_tol

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Airy_cpu_jacobian(x, mu, amp, offset, wl, NA, tol, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_mu(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_amp(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_offset(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_wl(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_NA(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1

            if bool2fit[5] :
                jacobian[model, point, count] = d_tol(x[point], mu[model], amp[model], offset[model], wl[model], NA[model], tol[model])
                count += 1
