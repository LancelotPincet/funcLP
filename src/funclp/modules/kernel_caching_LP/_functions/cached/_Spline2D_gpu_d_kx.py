
from ._Spline2D_gpukernel_d_kx import _Spline2D_gpukernel_d_kx as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Spline2D_gpu_d_kx(x, y, mux, muy, amp, offset, kx, ky, tx, ty, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], mux[model], muy[model], amp[model], offset[model], kx[model], ky[model], tx, ty, coeffs, )
