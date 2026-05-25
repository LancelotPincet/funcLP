
from funclp import ufunc
import numba as nb
from numba import cuda
_Spline2D_gpukernel_d_mux = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Spline2D_d_mux"])
