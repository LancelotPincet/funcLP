
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial4_gpukernel_function = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial4_function"])
