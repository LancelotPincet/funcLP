import numba as nb
from numba import cuda
from ._Normal_cpukernel_fisher import _Normal_cpukernel_fisher as kernel

@nb.njit(nogil=True, fastmath=True, cache=True)
def _LSE_Normal_cpukernel_fisher(raw_data, model_data, weights):
    return kernel(raw_data, model_data, weights, nb.float32(1), )
