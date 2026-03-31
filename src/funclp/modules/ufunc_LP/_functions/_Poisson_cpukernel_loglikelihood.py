
from funclp import ufunc
import numba as nb
_Poisson_cpukernel_loglikelihood = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Poisson_loglikelihood"])
