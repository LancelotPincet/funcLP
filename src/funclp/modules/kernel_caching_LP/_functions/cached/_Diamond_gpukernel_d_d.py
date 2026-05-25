
from funclp import ufunc
import numba as nb
from numba import cuda
_Diamond_gpukernel_d_d = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Diamond_d_d"])
