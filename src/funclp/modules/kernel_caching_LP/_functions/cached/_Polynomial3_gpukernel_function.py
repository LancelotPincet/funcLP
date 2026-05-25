
from funclp import ufunc
import numba as nb
from numba import cuda
_Polynomial3_gpukernel_function = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Polynomial3_function"])
