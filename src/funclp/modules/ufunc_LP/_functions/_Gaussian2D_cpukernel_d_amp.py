
from funclp import ufunc
import numba as nb
_Gaussian2D_cpukernel_d_amp = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gaussian2D_d_amp"])
