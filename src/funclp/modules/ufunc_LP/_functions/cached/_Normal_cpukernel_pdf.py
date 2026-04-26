
from funclp import ufunc
import numba as nb
_Normal_cpukernel_pdf = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Normal_pdf"])
