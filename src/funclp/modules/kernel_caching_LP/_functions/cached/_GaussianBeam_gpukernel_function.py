
from funclp import ufunc
import numba as nb
from numba import cuda
_GaussianBeam_gpukernel_function = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["GaussianBeam_function"])
