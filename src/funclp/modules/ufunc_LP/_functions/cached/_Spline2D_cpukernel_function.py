
from funclp import ufunc
import numba as nb
_Spline2D_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Spline2D_function"])
