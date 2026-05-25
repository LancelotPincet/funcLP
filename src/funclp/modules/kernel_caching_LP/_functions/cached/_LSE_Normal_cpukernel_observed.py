import numba as nb
from numba import cuda
from ._Normal_cpukernel_d2loglikelihood import _Normal_cpukernel_d2loglikelihood as kernel

@nb.njit(nogil=True, fastmath=True, cache=True)
def _LSE_Normal_cpukernel_observed(raw_data, model_data, weights):
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, nb.float32(1), )
