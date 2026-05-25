
from funclp import ufunc
import numba as nb
_Spline3D_cpukernel_d_kz = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Spline3D_d_kz"])
