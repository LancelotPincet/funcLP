import numba as nb
from numba import cuda
from ._Binomial_gpukernel_dloglikelihood import _Binomial_gpukernel_dloglikelihood as kernel

@nb.cuda.jit(device=True, fastmath=True, cache=True)
def _MLE_Binomial_gpukernel_loss(raw_data, model_data, weights):
    weights = (-1) * weights
    return kernel(raw_data, model_data, weights, np.int32(1), np.float32(1e-6), )
