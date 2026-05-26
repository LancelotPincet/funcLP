
from funclp import ufunc
import numba as nb
_Poisson_cpukernel_fisher = nb.njit(nogil=True, inline="always", fastmath=True, cache=True)(ufunc.main_functions["Poisson_fisher"])
