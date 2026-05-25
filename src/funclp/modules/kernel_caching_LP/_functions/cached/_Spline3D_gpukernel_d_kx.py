
from funclp import ufunc
import numba as nb
from numba import cuda
_Spline3D_gpukernel_d_kx = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Spline3D_d_kx"])
