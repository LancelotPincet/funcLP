
from funclp import ufunc
import numba as nb
_Airy2D_cpukernel_d_amp = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Airy2D_d_amp"])
