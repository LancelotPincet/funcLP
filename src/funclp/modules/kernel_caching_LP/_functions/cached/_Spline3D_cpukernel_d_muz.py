
from funclp import ufunc
import numba as nb
_Spline3D_cpukernel_d_muz = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Spline3D_d_muz"])
