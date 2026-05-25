from ._Polynomial3_gpukernel_d_a import _Polynomial3_gpukernel_d_a as d_a
from ._Polynomial3_gpukernel_d_b import _Polynomial3_gpukernel_d_b as d_b
from ._Polynomial3_gpukernel_d_c import _Polynomial3_gpukernel_d_c as d_c
from ._Polynomial3_gpukernel_d_d import _Polynomial3_gpukernel_d_d as d_d

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Polynomial3_gpu_jacobian(x, a, b, c, d, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_a(x[point], a[model], b[model], c[model], d[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_b(x[point], a[model], b[model], c[model], d[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_c(x[point], a[model], b[model], c[model], d[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_d(x[point], a[model], b[model], c[model], d[model])
            count += 1
