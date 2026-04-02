
from ._Gaussian2D_gpukernel_d_nsig import _Gaussian2D_gpukernel_d_nsig as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Gaussian2D_gpu_d_nsig(x, y, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model], )
