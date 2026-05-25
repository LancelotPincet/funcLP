
from funclp import ufunc
import numba as nb
from numba import cuda
_Airy_gpukernel_d_mu = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["Airy_d_mu"])
