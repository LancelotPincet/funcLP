import numba as nb
from numba import cuda
from ._Normal_gpukernel_d2loglikelihood import _Normal_gpukernel_d2loglikelihood as kernel

@nb.cuda.jit(device=True, fastmath=True, cache=True)
def _MLE_Normal_gpukernel_observed(raw_data, model_data, weights):
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, nb.float32(1), )
