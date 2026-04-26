
from ._Gaussian2D_cpukernel_d_theta import _Gaussian2D_cpukernel_d_theta as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Gaussian2D_cpu_d_theta(x, y, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], mux[model], muy[model], sigx[model], sigy[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], theta[model], )
