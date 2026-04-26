
from funclp import ufunc
import numba as nb
_Polynomial2_cpukernel_d_a = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial2_d_a"])
