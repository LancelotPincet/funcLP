from ._GaussianBeam_gpukernel_d_w0 import _GaussianBeam_gpukernel_d_w0 as d_w0
from ._GaussianBeam_gpukernel_d_z0 import _GaussianBeam_gpukernel_d_z0 as d_z0
from ._GaussianBeam_gpukernel_d_m2 import _GaussianBeam_gpukernel_d_m2 as d_m2
from ._GaussianBeam_gpukernel_d_wl import _GaussianBeam_gpukernel_d_wl as d_wl
from ._GaussianBeam_gpukernel_d_n import _GaussianBeam_gpukernel_d_n as d_n

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _GaussianBeam_gpu_jacobian(z, w0, z0, m2, wl, n, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_w0(z[point], w0[model], z0[model], m2[model], wl[model], n[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_z0(z[point], w0[model], z0[model], m2[model], wl[model], n[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_m2(z[point], w0[model], z0[model], m2[model], wl[model], n[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_wl(z[point], w0[model], z0[model], m2[model], wl[model], n[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_n(z[point], w0[model], z0[model], m2[model], wl[model], n[model])
            count += 1
