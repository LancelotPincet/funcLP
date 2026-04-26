
from ._Normal_gpukernel_dloglikelihood import _Normal_gpukernel_dloglikelihood as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _Normal_gpu_dloglikelihood(raw_data, model_data, weights, sigma, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(raw_data[model, point], model_data[model, point], weights[model, point], sigma[model], )
