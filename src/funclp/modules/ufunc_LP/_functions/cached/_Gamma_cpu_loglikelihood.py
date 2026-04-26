
from ._Gamma_cpukernel_loglikelihood import _Gamma_cpukernel_loglikelihood as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Gamma_cpu_loglikelihood(raw_data, model_data, weights, k, eps, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], k[model], eps[model], )
