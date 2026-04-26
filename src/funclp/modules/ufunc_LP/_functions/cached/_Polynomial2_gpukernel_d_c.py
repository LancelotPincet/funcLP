
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial2_gpukernel_d_c = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial2_d_c"])
