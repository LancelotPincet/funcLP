
from funclp import ufunc
import numba as nb
from numba import cuda
_Gamma_gpukernel_fisher = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Gamma_fisher"])
