
from funclp import ufunc
import numba as nb
from numba import cuda
_Rectangle_gpukernel_d_theta = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Rectangle_d_theta"])
