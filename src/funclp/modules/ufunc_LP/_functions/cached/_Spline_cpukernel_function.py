
from funclp import ufunc
import numba as nb
_Spline_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Spline_function"])
