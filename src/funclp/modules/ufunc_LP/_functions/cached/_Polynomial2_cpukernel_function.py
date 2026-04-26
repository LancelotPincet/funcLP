
from funclp import ufunc
import numba as nb
_Polynomial2_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial2_function"])
