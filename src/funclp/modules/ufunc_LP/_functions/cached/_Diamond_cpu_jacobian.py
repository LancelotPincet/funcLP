from ._Diamond_cpukernel_d_d import _Diamond_cpukernel_d_d as d_d
from ._Diamond_cpukernel_d_mux import _Diamond_cpukernel_d_mux as d_mux
from ._Diamond_cpukernel_d_muy import _Diamond_cpukernel_d_muy as d_muy
from ._Diamond_cpukernel_d_amp import _Diamond_cpukernel_d_amp as d_amp
from ._Diamond_cpukernel_d_offset import _Diamond_cpukernel_d_offset as d_offset

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Diamond_cpu_jacobian(x, y, d, mux, muy, amp, offset, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_d(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_mux(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model])
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_muy(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model])
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_amp(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model])
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_offset(x[point], y[point], d[model], mux[model], muy[model], amp[model], offset[model])
                count += 1
