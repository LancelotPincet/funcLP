
from funclp import ufunc
import numba as nb
from numba import cuda
_Gaussian_gpukernel_function = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Gaussian_function"])
