
from funclp import ufunc
import numba as nb
_Polynomial3_cpukernel_d_a = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial3_d_a"])
