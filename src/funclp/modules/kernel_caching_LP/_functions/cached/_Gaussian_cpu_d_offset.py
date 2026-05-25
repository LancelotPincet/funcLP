
from ._Gaussian_cpukernel_d_offset import _Gaussian_cpukernel_d_offset as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Gaussian_cpu_d_offset(x, mu, sig, amp, offset, pix, nsig, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], mu[model], sig[model], amp[model], offset[model], pix[model], nsig[model], )
