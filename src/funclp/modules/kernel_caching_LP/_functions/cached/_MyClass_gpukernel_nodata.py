
from funclp import ufunc
import numba as nb
from numba import cuda
_MyClass_gpukernel_nodata = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["MyClass_nodata"])
