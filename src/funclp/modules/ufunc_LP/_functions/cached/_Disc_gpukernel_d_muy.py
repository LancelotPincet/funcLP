
from funclp import ufunc
import numba as nb
from numba import cuda
_Disc_gpukernel_d_muy = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Disc_d_muy"])
