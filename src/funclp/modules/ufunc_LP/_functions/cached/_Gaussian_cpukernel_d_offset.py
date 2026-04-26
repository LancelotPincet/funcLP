
from funclp import ufunc
import numba as nb
_Gaussian_cpukernel_d_offset = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gaussian_d_offset"])
