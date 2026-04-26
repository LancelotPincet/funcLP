
from funclp import ufunc
import numba as nb
_Exponential2_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Exponential2_function"])
