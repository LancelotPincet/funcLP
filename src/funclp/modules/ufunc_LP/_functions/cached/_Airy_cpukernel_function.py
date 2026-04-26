
from funclp import ufunc
import numba as nb
_Airy_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Airy_function"])
