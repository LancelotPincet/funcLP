
from funclp import ufunc
import numba as nb
_Binomial_cpukernel_fisher = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Binomial_fisher"])
