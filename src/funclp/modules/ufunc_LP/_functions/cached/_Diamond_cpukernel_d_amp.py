
from funclp import ufunc
import numba as nb
_Diamond_cpukernel_d_amp = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Diamond_d_amp"])
