
from ._Gaussian_gpukernel_d_offset import _Gaussian_gpukernel_d_offset as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Gaussian_gpu_d_offset(x, mu, sig, amp, offset, pix, nsig, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model], )
