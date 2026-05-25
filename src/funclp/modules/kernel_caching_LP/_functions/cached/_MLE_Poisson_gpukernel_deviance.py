import numba as nb
from numba import cuda
from ._Poisson_gpukernel_loglikelihood import _Poisson_gpukernel_loglikelihood as kernel

@nb.cuda.jit(device=True, fastmath=True, cache=True)
def _MLE_Poisson_gpukernel_deviance(raw_data, model_data, weights):
    weights = (-2) * weights
    return kernel(raw_data, model_data, weights, nb.float32(1e-6), )
