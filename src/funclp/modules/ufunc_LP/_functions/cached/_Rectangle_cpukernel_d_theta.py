
from funclp import ufunc
import numba as nb
_Rectangle_cpukernel_d_theta = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Rectangle_d_theta"])
