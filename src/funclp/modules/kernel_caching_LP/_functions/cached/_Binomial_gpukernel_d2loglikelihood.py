
from funclp import ufunc
import numba as nb
from numba import cuda
_Binomial_gpukernel_d2loglikelihood = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Binomial_d2loglikelihood"])
