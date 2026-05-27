
from funclp import ufunc
import numba as nb
_Gaussian2D_cpukernel_d_pixx = nb.njit(nogil=True, inline="always", fastmath=False, cache=True)(ufunc.main_functions["Gaussian2D_d_pixx"])
