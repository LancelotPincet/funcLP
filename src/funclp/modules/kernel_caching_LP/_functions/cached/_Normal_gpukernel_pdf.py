
from funclp import ufunc
import numba as nb
from numba import cuda
_Normal_gpukernel_pdf = nb.cuda.jit(device=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Normal_pdf"])
