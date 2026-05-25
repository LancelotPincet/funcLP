
from ._Rectangle_gpukernel_d_l import _Rectangle_gpukernel_d_l as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Rectangle_gpu_d_l(x, y, l, ratio, mux, muy, amp, offset, theta, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], l[model], ratio[model], mux[model], muy[model], amp[model], offset[model], theta[model], )
