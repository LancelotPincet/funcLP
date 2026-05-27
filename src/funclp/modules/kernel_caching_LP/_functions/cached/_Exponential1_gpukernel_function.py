
from funclp import ufunc
import numba as nb
from numba import cuda
_Exponential1_gpukernel_function = nb.cuda.jit(device=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Exponential1_function"])
