
from funclp import ufunc
import numba as nb
from numba import cuda
_Binomial_gpukernel_dloglikelihood = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Binomial_dloglikelihood"])
