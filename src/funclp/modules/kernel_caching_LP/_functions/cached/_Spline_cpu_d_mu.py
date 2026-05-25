
from ._Spline_cpukernel_d_mu import _Spline_cpukernel_d_mu as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Spline_cpu_d_mu(x, mu, amp, offset, k, t, coeffs, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], mu[model], amp[model], offset[model], k[model], t, coeffs, )
