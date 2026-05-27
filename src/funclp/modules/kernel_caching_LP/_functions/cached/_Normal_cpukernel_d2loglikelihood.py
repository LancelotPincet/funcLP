
from funclp import ufunc
import numba as nb
_Normal_cpukernel_d2loglikelihood = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Normal_d2loglikelihood"])
