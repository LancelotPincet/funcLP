
from funclp import ufunc
import numba as nb
_Polynomial4_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial4_function"])
