
from funclp import ufunc
import numba as nb
from numba import cuda
_Exponential2_gpukernel_d_amp2 = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Exponential2_d_amp2"])
