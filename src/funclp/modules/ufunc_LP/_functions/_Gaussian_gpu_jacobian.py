from ._Gaussian_gpukernel_d_mu import _Gaussian_gpukernel_d_mu as d_mu
from ._Gaussian_gpukernel_d_sig import _Gaussian_gpukernel_d_sig as d_sig
from ._Gaussian_gpukernel_d_amp import _Gaussian_gpukernel_d_amp as d_amp
from ._Gaussian_gpukernel_d_offset import _Gaussian_gpukernel_d_offset as d_offset
from ._Gaussian_gpukernel_d_pix import _Gaussian_gpukernel_d_pix as d_pix
from ._Gaussian_gpukernel_d_nsig import _Gaussian_gpukernel_d_nsig as d_nsig

import numba as nb
from numba import cuda
@nb.cuda.jit()
def _Gaussian_gpu_jacobian(x, mu, sig, amp, offset, pix, nsig, jacobian, bool2fit, ignore) :
    nmodels, npoints, nparams = jacobian.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and not ignore[model] and point < npoints :
        count = 0

        if bool2fit[0] :
            jacobian[model, point, count] = d_mu(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1

        if bool2fit[1] :
            jacobian[model, point, count] = d_sig(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1

        if bool2fit[2] :
            jacobian[model, point, count] = d_amp(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1

        if bool2fit[3] :
            jacobian[model, point, count] = d_offset(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1

        if bool2fit[4] :
            jacobian[model, point, count] = d_pix(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1

        if bool2fit[5] :
            jacobian[model, point, count] = d_nsig(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model])
            count += 1
