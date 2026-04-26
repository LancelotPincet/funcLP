
from funclp import ufunc
import numba as nb
_Spline_cpukernel_d_mu = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Spline_d_mu"])
