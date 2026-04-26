
from funclp import ufunc
import numba as nb
_Polynomial5_cpukernel_d_d = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial5_d_d"])
