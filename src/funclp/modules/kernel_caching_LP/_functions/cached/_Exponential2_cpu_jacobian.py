from ._Exponential2_cpukernel_d_tau1 import _Exponential2_cpukernel_d_tau1 as d_tau1
from ._Exponential2_cpukernel_d_tau2 import _Exponential2_cpukernel_d_tau2 as d_tau2
from ._Exponential2_cpukernel_d_amp1 import _Exponential2_cpukernel_d_amp1 as d_amp1
from ._Exponential2_cpukernel_d_amp2 import _Exponential2_cpukernel_d_amp2 as d_amp2
from ._Exponential2_cpukernel_d_offset import _Exponential2_cpukernel_d_offset as d_offset

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Exponential2_cpu_jacobian(t, tau1, tau2, amp1, amp2, offset, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_tau1(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_tau2(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model])
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_amp1(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model])
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_amp2(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model])
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_offset(t[point], tau1[model], tau2[model], amp1[model], amp2[model], offset[model])
                count += 1
