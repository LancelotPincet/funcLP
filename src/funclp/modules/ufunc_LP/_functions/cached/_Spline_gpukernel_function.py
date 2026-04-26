
from funclp import ufunc
import numba as nb
from numba import cuda
_Spline_gpukernel_function = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Spline_function"])
