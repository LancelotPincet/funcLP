
from funclp import ufunc
import numba as nb
from numba import cuda
_GaussianBeam_gpukernel_d_m2 = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["GaussianBeam_d_m2"])
