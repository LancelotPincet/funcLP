
from funclp import ufunc
import numba as nb
from numba import cuda
_Poisson_gpukernel_loglikelihood_reduced = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Poisson_loglikelihood_reduced"])
