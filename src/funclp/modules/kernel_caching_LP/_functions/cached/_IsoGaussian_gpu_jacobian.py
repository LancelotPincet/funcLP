from ._IsoGaussian_gpukernel_d_mux import _IsoGaussian_gpukernel_d_mux as d_mux
from ._IsoGaussian_gpukernel_d_muy import _IsoGaussian_gpukernel_d_muy as d_muy
from ._IsoGaussian_gpukernel_d_sig import _IsoGaussian_gpukernel_d_sig as d_sig
from ._IsoGaussian_gpukernel_d_amp import _IsoGaussian_gpukernel_d_amp as d_amp
from ._IsoGaussian_gpukernel_d_offset import _IsoGaussian_gpukernel_d_offset as d_offset
from ._IsoGaussian_gpukernel_d_pixx import _IsoGaussian_gpukernel_d_pixx as d_pixx
from ._IsoGaussian_gpukernel_d_pixy import _IsoGaussian_gpukernel_d_pixy as d_pixy
from ._IsoGaussian_gpukernel_d_nsig import _IsoGaussian_gpukernel_d_nsig as d_nsig

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _IsoGaussian_gpu_jacobian(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_mux(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_muy(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_sig(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_amp(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_offset(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_pixx(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[6] :
            jacobian[model, point, count] = d_pixy(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1

        if bool2fit[7] :
            jacobian[model, point, count] = d_nsig(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model])
            count += 1
