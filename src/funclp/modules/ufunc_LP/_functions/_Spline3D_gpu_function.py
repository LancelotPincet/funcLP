
from ._Spline3D_gpukernel_function import _Spline3D_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Spline3D_gpu_function(x, y, z, mux, muy, muz, amp, offset, kx, ky, kz, tx, ty, tz, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], z[point], mux[model], muy[model], muz[model], amp[model], offset[model], kx[model], ky[model], kz[model], tx, ty, tz, coeffs, )
