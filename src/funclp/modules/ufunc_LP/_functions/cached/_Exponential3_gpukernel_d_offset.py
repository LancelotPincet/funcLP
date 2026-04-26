
from funclp import ufunc
import numba as nb
from numba import cuda
_Exponential3_gpukernel_d_offset = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Exponential3_d_offset"])
