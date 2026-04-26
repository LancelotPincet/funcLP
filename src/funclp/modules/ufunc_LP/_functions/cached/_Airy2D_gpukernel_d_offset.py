
from funclp import ufunc
import numba as nb
from numba import cuda
_Airy2D_gpukernel_d_offset = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Airy2D_d_offset"])
