
from ._Binomial_gpukernel_fisher import _Binomial_gpukernel_fisher as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Binomial_gpu_fisher(raw_data, model_data, weights, n, eps, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], n[model], eps[model], )
