from ._Polynomial4_cpukernel_d_a import _Polynomial4_cpukernel_d_a as d_a
from ._Polynomial4_cpukernel_d_b import _Polynomial4_cpukernel_d_b as d_b
from ._Polynomial4_cpukernel_d_c import _Polynomial4_cpukernel_d_c as d_c
from ._Polynomial4_cpukernel_d_d import _Polynomial4_cpukernel_d_d as d_d
from ._Polynomial4_cpukernel_d_e import _Polynomial4_cpukernel_d_e as d_e

import numba as nb
@nb.njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _Polynomial4_cpu_jacobian(x, a, b, c, d, e, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            count = 0

            if bool2fit[0] :
                jacobian[model, point, count] = d_a(x[point], a[model], b[model], c[model], d[model], e[model])
                count += 1

            if bool2fit[1] :
                jacobian[model, point, count] = d_b(x[point], a[model], b[model], c[model], d[model], e[model])
                count += 1

            if bool2fit[2] :
                jacobian[model, point, count] = d_c(x[point], a[model], b[model], c[model], d[model], e[model])
                count += 1

            if bool2fit[3] :
                jacobian[model, point, count] = d_d(x[point], a[model], b[model], c[model], d[model], e[model])
                count += 1

            if bool2fit[4] :
                jacobian[model, point, count] = d_e(x[point], a[model], b[model], c[model], d[model], e[model])
                count += 1
