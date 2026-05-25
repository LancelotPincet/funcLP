
from ._Normal_cpukernel_fisher import _Normal_cpukernel_fisher as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Normal_cpu_fisher(raw_data, model_data, weights, sigma, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], sigma[model], )
