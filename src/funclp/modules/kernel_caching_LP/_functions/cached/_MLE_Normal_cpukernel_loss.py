import numba as nb
from numba import cuda
from ._Normal_cpukernel_dloglikelihood import _Normal_cpukernel_dloglikelihood as kernel

@nb.njit(nogil=True, fastmath=True, cache=True)
def _MLE_Normal_cpukernel_loss(raw_data, model_data, weights):
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, nb.float32(1), )
