
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial1_gpukernel_d_a = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial1_d_a"])
