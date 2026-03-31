
from ._IsoGaussian_gpukernel_d_pixx import _IsoGaussian_gpukernel_d_pixx as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _IsoGaussian_gpu_d_pixx(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, eps, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], eps[model], )
