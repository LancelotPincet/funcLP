
from funclp import ufunc
import numba as nb
_Polynomial4_cpukernel_d_c = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial4_d_c"])
