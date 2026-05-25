
from funclp import ufunc
import numba as nb
from numba import cuda
_MyClass_gpukernel_novar = nb.cuda.jit(device=True, cache=True)(ufunc.main_functions["MyClass_novar"])
