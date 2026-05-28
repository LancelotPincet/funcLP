
from funclp import ufunc
import numba as nb
_Gaussian_cpukernel_function = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Gaussian_function"])
