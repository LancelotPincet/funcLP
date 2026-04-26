
from funclp import ufunc
import numba as nb
_Polynomial1_cpukernel_d_a = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial1_d_a"])
