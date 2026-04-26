
from funclp import ufunc
import numba as nb
_Spline2D_cpukernel_d_ky = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Spline2D_d_ky"])
