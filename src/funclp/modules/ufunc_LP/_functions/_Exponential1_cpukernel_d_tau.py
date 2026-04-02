
from funclp import ufunc
import numba as nb
_Exponential1_cpukernel_d_tau = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Exponential1_d_tau"])
