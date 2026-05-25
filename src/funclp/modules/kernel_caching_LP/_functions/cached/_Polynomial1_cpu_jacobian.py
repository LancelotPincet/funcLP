from ._Polynomial1_cpukernel_d_a import _Polynomial1_cpukernel_d_a as d_a
from ._Polynomial1_cpukernel_d_b import _Polynomial1_cpukernel_d_b as d_b

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Polynomial1_cpu_jacobian(x, a, b, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_a(x[point], a[model], b[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_b(x[point], a[model], b[model])
                count += 1
