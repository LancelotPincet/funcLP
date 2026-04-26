
from funclp import ufunc
import numba as nb
_Exponential3_cpukernel_d_tau2 = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Exponential3_d_tau2"])
