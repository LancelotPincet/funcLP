
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial5_gpukernel_d_b = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial5_d_b"])
