
from funclp import ufunc
import numba as nb
from numba import cuda
_IsoGaussian_gpukernel_d_amp = nb.cuda.jit(device=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["IsoGaussian_d_amp"])
