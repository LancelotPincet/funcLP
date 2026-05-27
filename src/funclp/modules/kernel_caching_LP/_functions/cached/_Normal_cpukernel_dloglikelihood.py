
from funclp import ufunc
import numba as nb
_Normal_cpukernel_dloglikelihood = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Normal_dloglikelihood"])
