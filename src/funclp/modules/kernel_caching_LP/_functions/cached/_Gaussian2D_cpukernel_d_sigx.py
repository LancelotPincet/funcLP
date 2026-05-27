
from funclp import ufunc
import numba as nb
_Gaussian2D_cpukernel_d_sigx = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Gaussian2D_d_sigx"])
