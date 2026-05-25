
from funclp import ufunc
import numba as nb
_Exponential2_cpukernel_d_tau2 = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Exponential2_d_tau2"])
