
from funclp import ufunc
import numba as nb
_Exponential3_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Exponential3_function"])
