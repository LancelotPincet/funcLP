
from funclp import ufunc
import numba as nb
_Diamond_cpukernel_d_d = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Diamond_d_d"])
