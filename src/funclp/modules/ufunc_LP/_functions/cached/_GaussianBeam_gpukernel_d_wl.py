
from funclp import ufunc
import numba as nb
from numba import cuda
_GaussianBeam_gpukernel_d_wl = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["GaussianBeam_d_wl"])
