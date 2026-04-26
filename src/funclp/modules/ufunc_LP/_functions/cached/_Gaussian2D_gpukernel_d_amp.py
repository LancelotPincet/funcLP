
from funclp import ufunc
import numba as nb
from numba import cuda
_Gaussian2D_gpukernel_d_amp = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Gaussian2D_d_amp"])
