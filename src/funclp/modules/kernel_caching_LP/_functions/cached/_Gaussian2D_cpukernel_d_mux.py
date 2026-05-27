
from funclp import ufunc
import numba as nb
_Gaussian2D_cpukernel_d_mux = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Gaussian2D_d_mux"])
