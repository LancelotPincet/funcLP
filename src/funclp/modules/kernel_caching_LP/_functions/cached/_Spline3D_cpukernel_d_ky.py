
from funclp import ufunc
import numba as nb
_Spline3D_cpukernel_d_ky = nb.njit(nogil=True, inline="always", fastmath=False, cache=True)(ufunc.main_functions["Spline3D_d_ky"])
