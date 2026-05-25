
from ._IsoGaussian_cpukernel_d_sig import _IsoGaussian_cpukernel_d_sig as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _IsoGaussian_cpu_d_sig(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], mux[model], muy[model], sig[model], amp[model], offset[model], pixx[model], pixy[model], nsig[model], )
