
from funclp import ufunc
import numba as nb
from numba import cuda
_Binomial_gpukernel_loglikelihood_reduced = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Binomial_loglikelihood_reduced"])
