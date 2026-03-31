
from funclp import ufunc
import numba as nb
from numba import cuda
_Poisson_gpukernel_dloglikelihood = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Poisson_dloglikelihood"])
