
from funclp import ufunc
import numba as nb
_Diamond_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Diamond_function"])
