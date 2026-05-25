
from ._Poisson_gpukernel_loglikelihood import _Poisson_gpukernel_loglikelihood as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Poisson_gpu_loglikelihood(raw_data, model_data, weights, eps, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], eps[model], )
