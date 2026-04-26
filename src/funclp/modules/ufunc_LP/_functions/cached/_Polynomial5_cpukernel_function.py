
from funclp import ufunc
import numba as nb
_Polynomial5_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Polynomial5_function"])
