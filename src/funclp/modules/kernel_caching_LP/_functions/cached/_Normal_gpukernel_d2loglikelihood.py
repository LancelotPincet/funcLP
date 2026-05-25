
from funclp import ufunc
import numba as nb
from numba import cuda
_Normal_gpukernel_d2loglikelihood = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Normal_d2loglikelihood"])
