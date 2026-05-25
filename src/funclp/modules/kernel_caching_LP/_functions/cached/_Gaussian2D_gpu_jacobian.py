from ._Gaussian2D_gpukernel_d_mux import _Gaussian2D_gpukernel_d_mux as d_mux
from ._Gaussian2D_gpukernel_d_muy import _Gaussian2D_gpukernel_d_muy as d_muy
from ._Gaussian2D_gpukernel_d_sigx import _Gaussian2D_gpukernel_d_sigx as d_sigx
from ._Gaussian2D_gpukernel_d_sigy import _Gaussian2D_gpukernel_d_sigy as d_sigy
from ._Gaussian2D_gpukernel_d_amp import _Gaussian2D_gpukernel_d_amp as d_amp
from ._Gaussian2D_gpukernel_d_offset import _Gaussian2D_gpukernel_d_offset as d_offset
from ._Gaussian2D_gpukernel_d_pixx import _Gaussian2D_gpukernel_d_pixx as d_pixx
from ._Gaussian2D_gpukernel_d_pixy import _Gaussian2D_gpukernel_d_pixy as d_pixy
from ._Gaussian2D_gpukernel_d_nsig import _Gaussian2D_gpukernel_d_nsig as d_nsig
from ._Gaussian2D_gpukernel_d_theta import _Gaussian2D_gpukernel_d_theta as d_theta

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Gaussian2D_gpu_jacobian(x, y, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_mux(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_muy(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_sigx(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_sigy(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_amp(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_offset(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[6] :
            jacobian[model, point, count] = d_pixx(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[7] :
            jacobian[model, point, count] = d_pixy(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[8] :
            jacobian[model, point, count] = d_nsig(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1

        if bool2fit[9] :
            jacobian[model, point, count] = d_theta(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model])
            count += 1
