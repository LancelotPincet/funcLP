
from funclp import ufunc
import numba as nb
_Airy_cpukernel_d_amp = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Airy_d_amp"])
