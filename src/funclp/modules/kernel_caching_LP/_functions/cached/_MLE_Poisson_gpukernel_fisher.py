import numba as nb
from numba import cuda
from ._Poisson_gpukernel_fisher import _Poisson_gpukernel_fisher as kernel

@nb.cuda.jit(device=True, fastmath=True, cache=True)
def _MLE_Poisson_gpukernel_fisher(raw_data, model_data, weights):
    return kernel(raw_data, model_data, weights, nb.float32(1e-6), )
