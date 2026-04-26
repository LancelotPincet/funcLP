
from funclp import ufunc
import numba as nb
_Disc_cpukernel_d_muy = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Disc_d_muy"])
