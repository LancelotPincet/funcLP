from ._Exponential1_cpukernel_d_tau import _Exponential1_cpukernel_d_tau as d_tau
from ._Exponential1_cpukernel_d_amp import _Exponential1_cpukernel_d_amp as d_amp
from ._Exponential1_cpukernel_d_offset import _Exponential1_cpukernel_d_offset as d_offset

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Exponential1_cpu_jacobian(t, tau, amp, offset, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_tau(t[point], tau[model], amp[model], offset[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_amp(t[point], tau[model], amp[model], offset[model])
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_offset(t[point], tau[model], amp[model], offset[model])
                count += 1
