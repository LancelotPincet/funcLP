import numba as nb
from numba import cuda
from ._Normal_gpukernel_dloglikelihood import _Normal_gpukernel_dloglikelihood as kernel

@nb.cuda.jit(device=True, fastmath=True, cache=True)
def _MLE_Normal_gpukernel_loss(raw_data, model_data, weights):
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, nb.float32(1), )
