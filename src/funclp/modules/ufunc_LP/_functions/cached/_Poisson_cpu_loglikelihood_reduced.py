
from ._Poisson_cpukernel_loglikelihood_reduced import _Poisson_cpukernel_loglikelihood_reduced as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _Poisson_cpu_loglikelihood_reduced(raw_data, model_data, weights, eps, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], eps[model], )
