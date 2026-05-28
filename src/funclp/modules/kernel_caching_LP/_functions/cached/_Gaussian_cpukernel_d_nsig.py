
from funclp import ufunc
import numba as nb
_Gaussian_cpukernel_d_nsig = nb.njit(nogil=True, inline="always", fastmath=False, cache=True)(ufunc.main_functions["Gaussian_d_nsig"])
