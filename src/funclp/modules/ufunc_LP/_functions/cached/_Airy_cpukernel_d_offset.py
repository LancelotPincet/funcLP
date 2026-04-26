
from funclp import ufunc
import numba as nb
_Airy_cpukernel_d_offset = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Airy_d_offset"])
