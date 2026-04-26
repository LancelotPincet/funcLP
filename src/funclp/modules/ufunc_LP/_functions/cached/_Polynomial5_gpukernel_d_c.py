
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial5_gpukernel_d_c = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial5_d_c"])
