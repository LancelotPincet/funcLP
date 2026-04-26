
from funclp import ufunc
import numba as nb
_Rectangle_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Rectangle_function"])
